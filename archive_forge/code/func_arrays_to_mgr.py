from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def arrays_to_mgr(arrays, columns: Index, index, *, dtype: DtypeObj | None=None, verify_integrity: bool=True, typ: str | None=None, consolidate: bool=True) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)
        arrays, refs = _homogenize(arrays, index, dtype)
    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        refs = [None] * len(arrays)
        for arr in arrays:
            if not isinstance(arr, (np.ndarray, ExtensionArray)) or arr.ndim != 1 or len(arr) != len(index):
                raise ValueError('Arrays must be 1-dimensional np.ndarray or ExtensionArray with length matching len(index)')
    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError('len(arrays) must match len(columns)')
    axes = [columns, index]
    if typ == 'block':
        return create_block_manager_from_column_arrays(arrays, axes, consolidate=consolidate, refs=refs)
    elif typ == 'array':
        return ArrayManager(arrays, [index, columns])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")