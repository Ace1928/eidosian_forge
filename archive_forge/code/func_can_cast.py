from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
import cupy as np
def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.

    See its docstring for more information.
    """
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f'from_={from_!r}, but should be an array_api array or dtype')
    if to not in _all_dtypes:
        raise TypeError(f'to={to!r}, but should be a dtype')
    try:
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        return False