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
def rec_array_to_mgr(data: np.rec.recarray | np.ndarray, index, columns, dtype: DtypeObj | None, copy: bool, typ: str) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)
    if columns is not None:
        columns = ensure_index(columns)
    arrays, arr_columns = to_arrays(fdata, columns)
    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns
    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ)
    if copy:
        mgr = mgr.copy()
    return mgr