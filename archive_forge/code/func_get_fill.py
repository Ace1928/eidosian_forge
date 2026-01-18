import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_fill(self):
    """Return whether face is colored."""
    return not cbook._str_lower_equal(self._original_facecolor, 'none')