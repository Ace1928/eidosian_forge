from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def first_color(colors):
    if colors.size == 0:
        return (0, 0, 0, 0)
    return tuple(colors[0])