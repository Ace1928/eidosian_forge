import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def get_topmost_subplotspec(self):
    """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
    gridspec = self.get_gridspec()
    if hasattr(gridspec, 'get_topmost_subplotspec'):
        return gridspec.get_topmost_subplotspec()
    else:
        return self