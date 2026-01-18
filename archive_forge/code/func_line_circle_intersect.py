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
def line_circle_intersect(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    dr2 = dx * dx + dy * dy
    D = x0 * y1 - x1 * y0
    D2 = D * D
    discrim = dr2 - D2
    if discrim >= 0.0:
        sign_dy = np.copysign(1, dy)
        sqrt_discrim = np.sqrt(discrim)
        return np.array([[(D * dy + sign_dy * dx * sqrt_discrim) / dr2, (-D * dx + abs(dy) * sqrt_discrim) / dr2], [(D * dy - sign_dy * dx * sqrt_discrim) / dr2, (-D * dx - abs(dy) * sqrt_discrim) / dr2]])
    else:
        return np.empty((0, 2))