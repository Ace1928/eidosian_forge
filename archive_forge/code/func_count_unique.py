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
def count_unique(data: np.ndarray, *, axis: int=-1) -> np.ndarray:
    """Count the number of unique values in a multi-dimensional array
    along a given axis.

    Parameters
    ----------
    data : np.ndarray
        The input array
    axis : int
        The target axis to count

    Returns
    -------
    n_uniques
        The number of unique values.
        This array will have one fewer dimension than the input.

    See Also
    --------
    is_unique

    Examples
    --------
    >>> x = np.vander(np.arange(5))
    >>> x
    array([[  0,   0,   0,   0,   1],
       [  1,   1,   1,   1,   1],
       [ 16,   8,   4,   2,   1],
       [ 81,  27,   9,   3,   1],
       [256,  64,  16,   4,   1]])
    >>> # Count unique values along rows (within columns)
    >>> librosa.util.count_unique(x, axis=0)
    array([5, 5, 5, 5, 1])
    >>> # Count unique values along columns (within rows)
    >>> librosa.util.count_unique(x, axis=-1)
    array([2, 1, 5, 5, 5])
    """
    return np.apply_along_axis(__count_unique, axis, data)