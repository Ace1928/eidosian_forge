import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@_docstring.interpd
def set_joinstyle(self, js):
    """
        Set the `.JoinStyle` for the collection (for all its elements).

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
    self._joinstyle = JoinStyle(js)