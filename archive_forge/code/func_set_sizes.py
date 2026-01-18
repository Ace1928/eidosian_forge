import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_sizes(self, sizes, dpi=72.0):
    """
        Set the sizes of each member of the collection.

        Parameters
        ----------
        sizes : `numpy.ndarray` or None
            The size to set for each element of the collection.  The
            value is the 'area' of the element.
        dpi : float, default: 72
            The dpi of the canvas.
        """
    if sizes is None:
        self._sizes = np.array([])
        self._transforms = np.empty((0, 3, 3))
    else:
        self._sizes = np.asarray(sizes)
        self._transforms = np.zeros((len(self._sizes), 3, 3))
        scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
        self._transforms[:, 0, 0] = scale
        self._transforms[:, 1, 1] = scale
        self._transforms[:, 2, 2] = 1.0
    self.stale = True