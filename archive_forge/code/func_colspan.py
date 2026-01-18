import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
@property
def colspan(self):
    """The columns spanned by this subplot, as a `range` object."""
    ncols = self.get_gridspec().ncols
    c1, c2 = sorted([self.num1 % ncols, self.num2 % ncols])
    return range(c1, c2 + 1)