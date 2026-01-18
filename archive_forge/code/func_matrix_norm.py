from __future__ import annotations
import functools
from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array
from .._core.internal import _normalize_axis_indices as normalize_axis_tuple
from typing import TYPE_CHECKING
from typing import NamedTuple
import cupy as np
def matrix_norm(x: Array, /, *, keepdims: bool=False, ord: Optional[Union[int, float, Literal['fro', 'nuc']]]='fro') -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in matrix_norm')
    return Array._new(np.linalg.norm(x._array, axis=(-2, -1), keepdims=keepdims, ord=ord))