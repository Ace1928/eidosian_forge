from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def infer_dtype_from_object(dtype) -> type:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    type
    """
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        try:
            _validate_date_like_dtype(dtype)
        except TypeError:
            pass
        if hasattr(dtype, 'numpy_dtype'):
            return dtype.numpy_dtype.type
        return dtype.type
    try:
        dtype = pandas_dtype(dtype)
    except TypeError:
        pass
    if isinstance(dtype, ExtensionDtype):
        return dtype.type
    elif isinstance(dtype, str):
        if dtype in ['datetimetz', 'datetime64tz']:
            return DatetimeTZDtype.type
        elif dtype in ['period']:
            raise NotImplementedError
        if dtype in ['datetime', 'timedelta']:
            dtype += '64'
        try:
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            pass
    return infer_dtype_from_object(np.dtype(dtype))