import warnings
from math import pi, sin, cos
import numpy as np
def draw_axis3d(ax, vector):
    ax.add_artist(Arrow3D([0, vector[0]], [0, vector[1]], [0, vector[2]], mutation_scale=20, arrowstyle='-|>', color='k'))