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
@rotation_point.setter
def rotation_point(self, value):
    if value in ['center', 'xy'] or (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Real) and isinstance(value[1], Real)):
        self._rotation_point = value
    else:
        raise ValueError("`rotation_point` must be one of {'xy', 'center', (number, number)}.")