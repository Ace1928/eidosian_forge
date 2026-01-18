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
def maybe_cast_pointwise_result(result: ArrayLike, dtype: DtypeObj, numeric_only: bool=False, same_dtype: bool=True) -> ArrayLike:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.

    Parameters
    ----------
    result : array-like
        Result to cast.
    dtype : np.dtype or ExtensionDtype
        Input Series from which result was calculated.
    numeric_only : bool, default False
        Whether to cast only numerics or datetimes as well.
    same_dtype : bool, default True
        Specify dtype when calling _from_sequence

    Returns
    -------
    result : array-like
        result maybe casted to the dtype.
    """
    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        if same_dtype:
            result = _maybe_cast_to_extension_array(cls, result, dtype=dtype)
        else:
            result = _maybe_cast_to_extension_array(cls, result)
    elif numeric_only and dtype.kind in 'iufcb' or not numeric_only:
        result = maybe_downcast_to_dtype(result, dtype)
    return result