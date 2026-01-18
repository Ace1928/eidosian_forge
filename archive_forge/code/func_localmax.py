from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def localmax(x: np.ndarray, *, axis: int=0) -> np.ndarray:
    """Find local maxima in an array

    An element ``x[i]`` is considered a local maximum if the following
    conditions are met:

    - ``x[i] > x[i-1]``
    - ``x[i] >= x[i+1]``

    Note that the first condition is strict, and that the first element
    ``x[0]`` will never be considered as a local maximum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmax(x, axis=0)
    array([[False, False, False],
           [ True, False, False],
           [False,  True,  True]], dtype=bool)
    >>> librosa.util.localmax(x, axis=1)
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True]], dtype=bool)

    Parameters
    ----------
    x : np.ndarray [shape=(d1,d2,...)]
        input vector or array
    axis : int
        axis along which to compute local maximality

    Returns
    -------
    m : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along ``axis``

    See Also
    --------
    localmin
    """
    xi = x.swapaxes(-1, axis)
    lmax = np.empty_like(x, dtype=bool)
    lmaxi = lmax.swapaxes(-1, axis)
    _localmax(xi, lmaxi)
    lmaxi[..., -1] = xi[..., -1] > xi[..., -2]
    return lmax