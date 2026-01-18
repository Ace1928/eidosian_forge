import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def data2grid(self, xd, yd):
    return (xd * self.x_data2grid, yd * self.y_data2grid)