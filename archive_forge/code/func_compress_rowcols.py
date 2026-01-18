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
def compress_rowcols(x, axis=None):
    """
    Suppress the rows and/or columns of a 2-D array that contain
    masked values.

    The suppression behavior is selected with the `axis` parameter.

    - If axis is None, both rows and columns are suppressed.
    - If axis is 0, only rows are suppressed.
    - If axis is 1 or -1, only columns are suppressed.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. Default is None.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    Examples
    --------
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> x
    masked_array(
      data=[[--, 1, 2],
            [--, 4, 5],
            [6, 7, 8]],
      mask=[[ True, False, False],
            [ True, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.compress_rowcols(x)
    array([[7, 8]])
    >>> np.ma.compress_rowcols(x, 0)
    array([[6, 7, 8]])
    >>> np.ma.compress_rowcols(x, 1)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    if asarray(x).ndim != 2:
        raise NotImplementedError('compress_rowcols works for 2D arrays only.')
    return compress_nd(x, axis=axis)