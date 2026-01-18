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
def set_xy2(self, x, y):
    """
        Set the *xy2* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
    if self._slope is None:
        self._xy2 = (x, y)
    else:
        raise ValueError("Cannot set an 'xy2' value while 'slope' is set; they differ but their functionalities overlap")