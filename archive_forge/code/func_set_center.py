import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
def set_center(self, xy):
    """
        Set the center of the annulus.

        Parameters
        ----------
        xy : (float, float)
        """
    self._center = xy
    self._path = None
    self.stale = True