import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_array(self):
    A = super().get_array()
    if A is None:
        return
    if self._deprecated_compression and np.any(np.ma.getmask(A)):
        _api.warn_deprecated('3.8', message='Getting the array from a PolyQuadMesh will return the full array in the future (uncompressed). To get this behavior now set the PolyQuadMesh with a 2D array .set_array(data2d).')
        return np.ma.compressed(A)
    return A