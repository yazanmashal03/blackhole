import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Physical Constants (Normalized for the simulation)
G = 1.0  # Gravitational constant
C = 10.0 # Speed of light (kept low for visual effect of relativity)

class BlackHole:
    def __init__(self, x, y, vx, vy, mass, color):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])
        self.mass = float(mass)
        self.color = color
        self.radius = 2 * G * self.mass / (C**2)
        self.path = []
        self.merged = False

    def update_radius(self):
        self.radius = 2 * G * self.mass / (C**2)

def calculate_acceleration(bh1, bh2, softening=0.1):
    r_vec = bh2.pos - bh1.pos
    distance = np.linalg.norm(r_vec)
    
    if distance < (bh1.radius + bh2.radius):
        return np.zeros(2), np.zeros(2), True # Collision detected
    
    force_mag = G * bh1.mass * bh2.mass / (distance**2 + softening**2)
    force_vec = force_mag * (r_vec / distance)
    
    accel1 = force_vec / bh1.mass
    accel2 = -force_vec / bh2.mass
    
    return accel1, accel2, False

# Simulation Setup
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.3)
ax.set_facecolor('black')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')

bh1_plot, = ax.plot([], [], 'o', color='blue', markersize=10, label='BH 1')
bh2_plot, = ax.plot([], [], 'o', color='red', markersize=10, label='BH 2')
bh1_path, = ax.plot([], [], '-', color='blue', alpha=0.3)
bh2_path, = ax.plot([], [], '-', color='red', alpha=0.3)

# Initial Parameters
init_mass1 = 5.0
init_mass2 = 5.0
init_dist = 10.0
init_v = 0.5

bh1 = BlackHole(-init_dist/2, 0, 0, -init_v, init_mass1, 'blue')
bh2 = BlackHole(init_dist/2, 0, 0, init_v, init_mass2, 'red')

# Sliders
ax_m1 = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_m2 = plt.axes([0.15, 0.16, 0.65, 0.03])
ax_dist = plt.axes([0.15, 0.12, 0.65, 0.03])
ax_v = plt.axes([0.15, 0.08, 0.65, 0.03])

s_m1 = Slider(ax_m1, 'Mass 1', 1.0, 20.0, valinit=init_mass1)
s_m2 = Slider(ax_m2, 'Mass 2', 1.0, 20.0, valinit=init_mass2)
s_dist = Slider(ax_dist, 'Distance', 2.0, 20.0, valinit=init_dist)
s_v = Slider(ax_v, 'Initial V', 0.0, 2.0, valinit=init_v)

reset_ax = plt.axes([0.85, 0.08, 0.1, 0.04])
button_reset = Button(reset_ax, 'Reset', color='gray', hovercolor='0.975')

pause_ax = plt.axes([0.85, 0.13, 0.1, 0.04])
button_pause = Button(pause_ax, 'Pause', color='gray', hovercolor='0.975')

is_merged = False
is_paused = False

def init():
    bh1_plot.set_data([], [])
    bh2_plot.set_data([], [])
    bh1_path.set_data([], [])
    bh2_path.set_data([], [])
    return bh1_plot, bh2_plot, bh1_path, bh2_path

def update(frame):
    global bh1, bh2, is_merged, is_paused
    
    if is_paused:
        return bh1_plot, bh2_plot, bh1_path, bh2_path

    if not is_merged:
        # Physics Step
        accel1, accel2, collision = calculate_acceleration(bh1, bh2)
        
        if collision:
            is_merged = True
            # Simple conservation of momentum for merger
            new_mass = bh1.mass + bh2.mass
            new_vel = (bh1.mass * bh1.vel + bh2.mass * bh2.vel) / new_mass
            new_pos = (bh1.mass * bh1.pos + bh2.mass * bh2.pos) / new_mass
            
            bh1.mass = new_mass
            bh1.vel = new_vel
            bh1.pos = new_pos
            bh1.update_radius()
            bh2.merged = True
        else:
            bh1.vel += accel1 * 0.1 # dt = 0.1
            bh1.pos += bh1.vel * 0.1
            bh2.vel += accel2 * 0.1
            bh2.pos += bh2.vel * 0.1
            
            bh1.path.append(bh1.pos.copy())
            bh2.path.append(bh2.pos.copy())

    # Update plots
    bh1_plot.set_data([bh1.pos[0]], [bh1.pos[1]])
    # Use markersize proportional to radius, but with a minimum
    ms1 = max(bh1.radius * 5, 2)
    bh1_plot.set_markersize(ms1)
    
    p1 = np.array(bh1.path)
    if len(p1) > 0:
        bh1_path.set_data(p1[:, 0], p1[:, 1])

    if not bh2.merged:
        bh2_plot.set_data([bh2.pos[0]], [bh2.pos[1]])
        ms2 = max(bh2.radius * 5, 2)
        bh2_plot.set_markersize(ms2)
        p2 = np.array(bh2.path)
        if len(p2) > 0:
            bh2_path.set_data(p2[:, 0], p2[:, 1])
    else:
        bh2_plot.set_data([], [])
        bh2_path.set_data([], [])

    return bh1_plot, bh2_plot, bh1_path, bh2_path

def reset(event):
    global bh1, bh2, is_merged
    bh1 = BlackHole(-s_dist.val/2, 0, 0, -s_v.val, s_m1.val, 'blue')
    bh2 = BlackHole(s_dist.val/2, 0, 0, s_v.val, s_m2.val, 'red')
    is_merged = False

def toggle_pause(event):
    global is_paused
    is_paused = not is_paused
    button_pause.label.set_text('Resume' if is_paused else 'Pause')

button_reset.on_clicked(reset)
button_pause.on_clicked(toggle_pause)

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=20)
plt.title("Black Hole Collision Simulation")
plt.show()
