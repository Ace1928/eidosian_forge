import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_offset_transform(self, offset_transform):
    """
        Set the artist offset transform.

        Parameters
        ----------
        offset_transform : `.Transform`
        """
    self._offset_transform = offset_transform