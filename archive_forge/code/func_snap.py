from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import (
import pandas.core.common as com
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
def snap(self, freq: Frequency='S') -> DatetimeIndex:
    """
        Snap time stamps to nearest occurring frequency.

        Returns
        -------
        DatetimeIndex

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
    freq = to_offset(freq)
    dta = self._data.copy()
    for i, v in enumerate(self):
        s = v
        if not freq.is_on_offset(s):
            t0 = freq.rollback(s)
            t1 = freq.rollforward(s)
            if abs(s - t0) < abs(t1 - s):
                s = t0
            else:
                s = t1
        dta[i] = s
    return DatetimeIndex._simple_new(dta, name=self.name)