import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def interpolated(self, steps):
    """
        Return a new path resampled to length N x *steps*.

        Codes other than `LINETO` are not handled correctly.
        """
    if steps == 1:
        return self
    vertices = simple_linear_interpolation(self.vertices, steps)
    codes = self.codes
    if codes is not None:
        new_codes = np.full((len(codes) - 1) * steps + 1, Path.LINETO, dtype=self.code_type)
        new_codes[0::steps] = codes
    else:
        new_codes = None
    return Path(vertices, new_codes)