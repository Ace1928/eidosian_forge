from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def set_nulls(data: np.ndarray | pd.Series, col: Column, validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None, allow_modify_inplace: bool=True):
    """
    Set null values for the data according to the column null kind.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to set nulls in.
    col : Column
        Column object that describes the `data`.
    validity : tuple(Buffer, dtype) or None
        The return value of ``col.buffers()``. We do not access the ``col.buffers()``
        here to not take the ownership of the memory of buffer objects.
    allow_modify_inplace : bool, default: True
        Whether to modify the `data` inplace when zero-copy is possible (True) or always
        modify a copy of the `data` (False).

    Returns
    -------
    np.ndarray or pd.Series
        Data with the nulls being set.
    """
    null_kind, sentinel_val = col.describe_null
    null_pos = None
    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = pd.Series(data) == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert validity, 'Expected to have a validity buffer for the mask'
        valid_buff, valid_dtype = validity
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, offset=col.offset, length=col.size())
        if sentinel_val == 0:
            null_pos = ~null_pos
    elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
        pass
    else:
        raise NotImplementedError(f'Null kind {null_kind} is not yet supported.')
    if null_pos is not None and np.any(null_pos):
        if not allow_modify_inplace:
            data = data.copy()
        try:
            data[null_pos] = None
        except TypeError:
            data = data.astype(float)
            data[null_pos] = None
        except SettingWithCopyError:
            data = data.copy()
            data[null_pos] = None
    return data