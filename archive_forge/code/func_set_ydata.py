import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def set_ydata(self, y):
    """
        Set the data array for y.

        Parameters
        ----------
        y : 1D array
        """
    if not np.iterable(y):
        _api.warn_deprecated(since='3.7', message='Setting data with a non sequence type is deprecated since %(since)s and will be remove %(removal)s')
        y = [y]
    self._yorig = copy.copy(y)
    self._invalidy = True
    self.stale = True