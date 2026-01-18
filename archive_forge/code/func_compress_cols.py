import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def compress_cols(a):
    """
    Suppress whole columns of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
    `compress_rowcols` for details.

    See Also
    --------
    compress_rowcols

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError('compress_cols works for 2D arrays only.')
    return compress_rowcols(a, 1)