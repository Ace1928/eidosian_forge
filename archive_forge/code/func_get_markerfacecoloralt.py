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
def get_markerfacecoloralt(self):
    """
        Return the alternate marker face color.

        See also `~.Line2D.set_markerfacecoloralt`.
        """
    return self._get_markerfacecolor(alt=True)