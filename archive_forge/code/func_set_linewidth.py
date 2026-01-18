import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_linewidth(self, lw):
    """
        Set the linewidth(s) for the collection.  *lw* can be a scalar
        or a sequence; if it is a sequence the patches will cycle
        through the sequence

        Parameters
        ----------
        lw : float or list of floats
        """
    if lw is None:
        lw = self._get_default_linewidth()
    self._us_lw = np.atleast_1d(lw)
    self._linewidths, self._linestyles = self._bcast_lwls(self._us_lw, self._us_linestyles)
    self.stale = True