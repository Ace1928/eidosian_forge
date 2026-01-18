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
@staticmethod
def ensure_quadratic_bezier(path):
    """
            Some ArrowStyle classes only works with a simple quadratic
            Bézier curve (created with `.ConnectionStyle.Arc3` or
            `.ConnectionStyle.Angle3`). This static method checks if the
            provided path is a simple quadratic Bézier curve and returns its
            control points if true.
            """
    segments = list(path.iter_segments())
    if len(segments) != 2 or segments[0][1] != Path.MOVETO or segments[1][1] != Path.CURVE3:
        raise ValueError("'path' is not a valid quadratic Bezier curve")
    return [*segments[0][0], *segments[1][0]]