import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def set_height_ratios(self, height_ratios):
    """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
    if height_ratios is None:
        height_ratios = [1] * self._nrows
    elif len(height_ratios) != self._nrows:
        raise ValueError('Expected the given number of height ratios to match the number of rows of the grid')
    self._row_height_ratios = height_ratios