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
def set_slope(self, slope):
    """
        Set the *slope* value of the line.

        Parameters
        ----------
        slope : float
            The slope of the line.
        """
    if self._xy2 is None:
        self._slope = slope
    else:
        raise ValueError("Cannot set a 'slope' value while 'xy2' is set; they differ but their functionalities overlap")