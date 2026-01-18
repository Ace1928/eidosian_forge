import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_linelength(self):
    """Return the length of the lines used to mark each event."""
    return self._linelength