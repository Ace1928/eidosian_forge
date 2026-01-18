import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def reset_start_point(self, xg, yg):
    xm, ym = self.grid2mask(xg, yg)
    self.mask._current_xy = (xm, ym)