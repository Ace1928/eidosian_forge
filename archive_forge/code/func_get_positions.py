import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_positions(self):
    """
        Return an array containing the floating-point values of the positions.
        """
    pos = 0 if self.is_horizontal() else 1
    return [segment[0, pos] for segment in self.get_segments()]