from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == 'M':
            from pandas.core.arrays import DatetimeArray
            dtype = get_supported_dtype(arr.dtype)
            return DatetimeArray._from_sequence(arr, dtype=dtype)
        elif arr.dtype.kind == 'm':
            from pandas.core.arrays import TimedeltaArray
            dtype = get_supported_dtype(arr.dtype)
            return TimedeltaArray._from_sequence(arr, dtype=dtype)
    return arr