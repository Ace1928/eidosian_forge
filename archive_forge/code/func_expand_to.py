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
def expand_to(x: np.ndarray, *, ndim: int, axes: Union[int, slice, Sequence[int], Sequence[slice]]) -> np.ndarray:
    """Expand the dimensions of an input array with

    Parameters
    ----------
    x : np.ndarray
        The input array
    ndim : int
        The number of dimensions to expand to.  Must be at least ``x.ndim``
    axes : int or slice
        The target axis or axes to preserve from x.
        All other axes will have length 1.

    Returns
    -------
    x_exp : np.ndarray
        The expanded version of ``x``, satisfying the following:
            ``x_exp[axes] == x``
            ``x_exp.ndim == ndim``

    See Also
    --------
    np.expand_dims

    Examples
    --------
    Expand a 1d array into an (n, 1) shape

    >>> x = np.arange(3)
    >>> librosa.util.expand_to(x, ndim=2, axes=0)
    array([[0],
       [1],
       [2]])

    Expand a 1d array into a (1, n) shape

    >>> librosa.util.expand_to(x, ndim=2, axes=1)
    array([[0, 1, 2]])

    Expand a 2d array into (1, n, m, 1) shape

    >>> x = np.vander(np.arange(3))
    >>> librosa.util.expand_to(x, ndim=4, axes=[1,2]).shape
    (1, 3, 3, 1)
    """
    axes_tup: Tuple[int]
    try:
        axes_tup = tuple(axes)
    except TypeError:
        axes_tup = tuple([axes])
    if len(axes_tup) != x.ndim:
        raise ParameterError(f'Shape mismatch between axes={axes_tup} and input x.shape={x.shape}')
    if ndim < x.ndim:
        raise ParameterError(f'Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}')
    shape: List[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]
    return x.reshape(shape)