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
def indexer_between_time(self, start_time, end_time, include_start: bool=True, include_end: bool=True) -> npt.NDArray[np.intp]:
    """
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
                           '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
                          dtype='datetime64[ns]', freq='h')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        """
    start_time = to_time(start_time)
    end_time = to_time(end_time)
    time_micros = self._get_time_micros()
    start_micros = _time_to_micros(start_time)
    end_micros = _time_to_micros(end_time)
    if include_start and include_end:
        lop = rop = operator.le
    elif include_start:
        lop = operator.le
        rop = operator.lt
    elif include_end:
        lop = operator.lt
        rop = operator.le
    else:
        lop = rop = operator.lt
    if start_time <= end_time:
        join_op = operator.and_
    else:
        join_op = operator.or_
    mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))
    return mask.nonzero()[0]