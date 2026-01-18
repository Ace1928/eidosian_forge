import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_antialiased(self):
    """
        Get the antialiasing state for rendering.

        Returns
        -------
        array of bools
        """
    return self._antialiaseds