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
def compress_nd(x, axis=None):
    """Suppress slices from multiple dimensions which contain masked values.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with `mask`
        set to `nomask`.
    axis : tuple of ints or int, optional
        Which dimensions to suppress slices from can be configured with this
        parameter.
        - If axis is a tuple of ints, those are the axes to suppress slices from.
        - If axis is an int, then that is the only axis to suppress slices from.
        - If axis is None, all axis are selected.

    Returns
    -------
    compress_array : ndarray
        The compressed array.
    """
    x = asarray(x)
    m = getmask(x)
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = normalize_axis_tuple(axis, x.ndim)
    if m is nomask or not m.any():
        return x._data
    if m.all():
        return nxarray([])
    data = x._data
    for ax in axis:
        axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
        data = data[(slice(None),) * ax + (~m.any(axis=axes),)]
    return data