import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def grid2data(self, xg, yg):
    return (xg / self.x_data2grid, yg / self.y_data2grid)