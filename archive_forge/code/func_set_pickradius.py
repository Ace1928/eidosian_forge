import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_pickradius(self, pickradius):
    """
        Set the pick radius used for containment tests.

        Parameters
        ----------
        pickradius : float
            Pick radius, in points.
        """
    if not isinstance(pickradius, Real):
        raise ValueError(f'pickradius must be a real-valued number, not {pickradius!r}')
    self._pickradius = pickradius