from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import (
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype('object')
    elif dtype == np.dtype(str):
        dtype = np.dtype('object')
    return dtype