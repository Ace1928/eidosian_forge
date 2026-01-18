from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def maybe_convert_platform(values: list | tuple | range | np.ndarray | ExtensionArray) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
    arr: ArrayLike
    if isinstance(values, (list, tuple, range)):
        arr = construct_1d_object_array_from_listlike(values)
    else:
        arr = values
    if arr.dtype == _dtype_obj:
        arr = cast(np.ndarray, arr)
        arr = lib.maybe_convert_objects(arr)
    return arr