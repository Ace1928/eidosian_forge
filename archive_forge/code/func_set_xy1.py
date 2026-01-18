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
def set_xy1(self, x, y):
    """
        Set the *xy1* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
    self._xy1 = (x, y)