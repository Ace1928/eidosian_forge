import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def new_subplotspec(self, loc, rowspan=1, colspan=1):
    """
        Create and return a `.SubplotSpec` instance.

        Parameters
        ----------
        loc : (int, int)
            The position of the subplot in the grid as
            ``(row_index, column_index)``.
        rowspan, colspan : int, default: 1
            The number of rows and columns the subplot should span in the grid.
        """
    loc1, loc2 = loc
    subplotspec = self[loc1:loc1 + rowspan, loc2:loc2 + colspan]
    return subplotspec