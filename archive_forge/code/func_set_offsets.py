import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_offsets(self, offsets):
    """
        Set the offsets for the collection.

        Parameters
        ----------
        offsets : (N, 2) or (2,) array-like
        """
    offsets = np.asanyarray(offsets)
    if offsets.shape == (2,):
        offsets = offsets[None, :]
    cstack = np.ma.column_stack if isinstance(offsets, np.ma.MaskedArray) else np.column_stack
    self._offsets = cstack((np.asanyarray(self.convert_xunits(offsets[:, 0]), float), np.asanyarray(self.convert_yunits(offsets[:, 1]), float)))
    self.stale = True