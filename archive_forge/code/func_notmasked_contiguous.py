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
def notmasked_contiguous(a, axis=None):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array, and this
        is the same as `flatnotmasked_contiguous`.

    Returns
    -------
    endpoints : list
        A list of slices (start and end indexes) of unmasked indexes
        in the array.

        If the input is 2d and axis is specified, the result is a list of lists.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> a = np.arange(12).reshape((3, 4))
    >>> mask = np.zeros_like(a)
    >>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
    >>> ma = np.ma.array(a, mask=mask)
    >>> ma
    masked_array(
      data=[[0, --, 2, 3],
            [--, --, --, 7],
            [8, --, --, 11]],
      mask=[[False,  True, False, False],
            [ True,  True,  True, False],
            [False,  True,  True, False]],
      fill_value=999999)
    >>> np.array(ma[~ma.mask])
    array([ 0,  2,  3,  7, 8, 11])

    >>> np.ma.notmasked_contiguous(ma)
    [slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]

    >>> np.ma.notmasked_contiguous(ma, axis=0)
    [[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]

    >>> np.ma.notmasked_contiguous(ma, axis=1)
    [[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]

    """
    a = asarray(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError('Currently limited to at most 2D array.')
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    result = []
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result