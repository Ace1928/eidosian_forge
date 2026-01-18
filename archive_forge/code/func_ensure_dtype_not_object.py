from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def ensure_dtype_not_object(var: Variable, name: T_Name=None) -> Variable:
    if var.dtype.kind == 'O':
        dims, data, attrs, encoding = variables.unpack_for_encoding(var)
        if strings.check_vlen_dtype(data.dtype) is not None:
            return var
        if is_duck_dask_array(data):
            emit_user_level_warning(f'variable {name} has data in the form of a dask array with dtype=object, which means it is being loaded into memory to determine a data type that can be safely stored on disk. To avoid this, coerce this variable to a fixed-size dtype with astype() before saving it.', category=SerializationWarning)
            data = data.compute()
        missing = pd.isnull(data)
        if missing.any():
            non_missing_values = data[~missing]
            inferred_dtype = _infer_dtype(non_missing_values, name)
            fill_value: bytes | str
            if strings.is_bytes_dtype(inferred_dtype):
                fill_value = b''
            elif strings.is_unicode_dtype(inferred_dtype):
                fill_value = ''
            else:
                if not np.issubdtype(inferred_dtype, np.floating):
                    inferred_dtype = np.dtype(float)
                fill_value = inferred_dtype.type(np.nan)
            data = _copy_with_dtype(data, dtype=inferred_dtype)
            data[missing] = fill_value
        else:
            data = _copy_with_dtype(data, dtype=_infer_dtype(data, name))
        assert data.dtype.kind != 'O' or data.dtype.metadata
        var = Variable(dims, data, attrs, encoding, fastpath=True)
    return var