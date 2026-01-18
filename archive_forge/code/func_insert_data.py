from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
    if self.index is not None:
        temp = self.frame.copy()
        temp.index.names = self.index
        try:
            temp.reset_index(inplace=True)
        except ValueError as err:
            raise ValueError(f'duplicate name in index/columns: {err}') from err
    else:
        temp = self.frame
    column_names = list(map(str, temp.columns))
    ncols = len(column_names)
    data_list: list[np.ndarray] = [None] * ncols
    for i, (_, ser) in enumerate(temp.items()):
        if ser.dtype.kind == 'M':
            if isinstance(ser._values, ArrowExtensionArray):
                import pyarrow as pa
                if pa.types.is_date(ser.dtype.pyarrow_dtype):
                    d = ser._values.to_numpy(dtype=object)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=FutureWarning)
                        d = np.asarray(ser.dt.to_pydatetime(), dtype=object)
            else:
                d = ser._values.to_pydatetime()
        elif ser.dtype.kind == 'm':
            vals = ser._values
            if isinstance(vals, ArrowExtensionArray):
                vals = vals.to_numpy(dtype=np.dtype('m8[ns]'))
            d = vals.view('i8').astype(object)
        else:
            d = ser._values.astype(object)
        assert isinstance(d, np.ndarray), type(d)
        if ser._can_hold_na:
            mask = isna(d)
            d[mask] = None
        data_list[i] = d
    return (column_names, data_list)