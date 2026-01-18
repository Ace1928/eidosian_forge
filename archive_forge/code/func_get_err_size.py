from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def get_err_size(self, legend, xdescent, ydescent, width, height, fontsize):
    xerr_size = self._xerr_size * fontsize
    if self._yerr_size is None:
        yerr_size = xerr_size
    else:
        yerr_size = self._yerr_size * fontsize
    return (xerr_size, yerr_size)