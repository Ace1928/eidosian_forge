from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def na_value_for_dtype(dtype: DtypeObj, compat: bool=True):
    """
    Return a dtype compat na value

    Parameters
    ----------
    dtype : string / dtype
    compat : bool, default True

    Returns
    -------
    np.dtype or a pandas dtype

    Examples
    --------
    >>> na_value_for_dtype(np.dtype('int64'))
    0
    >>> na_value_for_dtype(np.dtype('int64'), compat=False)
    nan
    >>> na_value_for_dtype(np.dtype('float64'))
    nan
    >>> na_value_for_dtype(np.dtype('bool'))
    False
    >>> na_value_for_dtype(np.dtype('datetime64[ns]'))
    numpy.datetime64('NaT')
    """
    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in 'mM':
        unit = np.datetime_data(dtype)[0]
        return dtype.type('NaT', unit)
    elif dtype.kind == 'f':
        return np.nan
    elif dtype.kind in 'iu':
        if compat:
            return 0
        return np.nan
    elif dtype.kind == 'b':
        if compat:
            return False
        return np.nan
    return np.nan