import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def set_vertical(self, v):
    """
        Parameters
        ----------
        v : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            sizes for vertical division
        """
    self._vertical = v