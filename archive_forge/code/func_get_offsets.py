import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_offsets(self):
    """Return the offsets for the collection."""
    return np.zeros((1, 2)) if self._offsets is None else self._offsets