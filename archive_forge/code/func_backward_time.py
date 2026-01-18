import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def backward_time(xi, yi):
    dxi, dyi = forward_time(xi, yi)
    return (-dxi, -dyi)