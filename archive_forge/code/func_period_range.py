from __future__ import annotations
from datetime import (
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import (
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import inherit_names
def period_range(start=None, end=None, periods: int | None=None, freq=None, name: Hashable | None=None) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if freq is None and (not isinstance(start, Period) and (not isinstance(end, Period))):
        freq = 'D'
    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)