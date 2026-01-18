import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def get_vertical_sizes(self, renderer):
    return np.array([s.get_size(renderer) for s in self.get_vertical()])