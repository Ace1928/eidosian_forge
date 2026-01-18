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
def maybe_box_native(value: Scalar | None | NAType) -> Scalar | None | NAType:
    """
    If passed a scalar cast the scalar to a python native type.

    Parameters
    ----------
    value : scalar or Series

    Returns
    -------
    scalar or Series
    """
    if is_float(value):
        value = float(value)
    elif is_integer(value):
        value = int(value)
    elif is_bool(value):
        value = bool(value)
    elif isinstance(value, (np.datetime64, np.timedelta64)):
        value = maybe_box_datetimelike(value)
    elif value is NA:
        value = None
    return value