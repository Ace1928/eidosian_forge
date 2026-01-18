import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def mask2grid(self, xm, ym):
    return (xm * self.x_mask2grid, ym * self.y_mask2grid)