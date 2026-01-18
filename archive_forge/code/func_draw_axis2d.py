import warnings
from math import pi, sin, cos
import numpy as np
def draw_axis2d(ax, x, y):
    ax.arrow(0, 0, x, y, lw=1, color='k', length_includes_head=True, head_width=0.03, head_length=0.05)