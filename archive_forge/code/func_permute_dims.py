from __future__ import annotations
from ._array_object import Array
from ._data_type_functions import result_type
from typing import List, Optional, Tuple, Union
import cupy as np
def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return Array._new(np.transpose(x._array, axes))