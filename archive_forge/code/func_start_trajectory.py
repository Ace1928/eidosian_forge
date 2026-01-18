import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def start_trajectory(self, xg, yg, broken_streamlines=True):
    xm, ym = self.grid2mask(xg, yg)
    self.mask._start_trajectory(xm, ym, broken_streamlines)