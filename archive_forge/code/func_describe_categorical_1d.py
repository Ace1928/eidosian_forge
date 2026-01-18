from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
def describe_categorical_1d(data: Series, percentiles_ignored: Sequence[float]) -> Series:
    """Describe series containing categorical data.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ['count', 'unique', 'top', 'freq']
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    if count_unique > 0:
        top, freq = (objcounts.index[0], objcounts.iloc[0])
        dtype = None
    else:
        top, freq = (np.nan, np.nan)
        dtype = 'object'
    result = [data.count(), count_unique, top, freq]
    from pandas import Series
    return Series(result, index=names, name=data.name, dtype=dtype)