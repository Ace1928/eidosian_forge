import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def is_horizontal(self):
    """True if the eventcollection is horizontal, False if vertical."""
    return self._is_horizontal