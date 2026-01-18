from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
@pytest.mark.parametrize('timezone', [pytest.param(None), pytest.param('Europe/Berlin', marks=pytest.mark.skipif(StorageFormat.get() == 'Hdk', reason='HDK is unable to store TZ in the table schema'))])
def test_dt(timezone):
    data = pd.date_range('2016-12-31', periods=128, freq='D', tz=timezone)
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)
    df_equals(modin_series.dt.date, pandas_series.dt.date)
    df_equals(modin_series.dt.time, pandas_series.dt.time)
    df_equals(modin_series.dt.timetz, pandas_series.dt.timetz)
    df_equals(modin_series.dt.year, pandas_series.dt.year)
    df_equals(modin_series.dt.month, pandas_series.dt.month)
    df_equals(modin_series.dt.day, pandas_series.dt.day)
    df_equals(modin_series.dt.hour, pandas_series.dt.hour)
    df_equals(modin_series.dt.minute, pandas_series.dt.minute)
    df_equals(modin_series.dt.second, pandas_series.dt.second)
    df_equals(modin_series.dt.microsecond, pandas_series.dt.microsecond)
    df_equals(modin_series.dt.nanosecond, pandas_series.dt.nanosecond)
    df_equals(modin_series.dt.dayofweek, pandas_series.dt.dayofweek)
    df_equals(modin_series.dt.day_of_week, pandas_series.dt.day_of_week)
    df_equals(modin_series.dt.weekday, pandas_series.dt.weekday)
    df_equals(modin_series.dt.dayofyear, pandas_series.dt.dayofyear)
    df_equals(modin_series.dt.day_of_year, pandas_series.dt.day_of_year)
    df_equals(modin_series.dt.unit, pandas_series.dt.unit)
    df_equals(modin_series.dt.as_unit('s'), pandas_series.dt.as_unit('s'))
    df_equals(modin_series.dt.isocalendar(), pandas_series.dt.isocalendar())
    df_equals(modin_series.dt.quarter, pandas_series.dt.quarter)
    df_equals(modin_series.dt.is_month_start, pandas_series.dt.is_month_start)
    df_equals(modin_series.dt.is_month_end, pandas_series.dt.is_month_end)
    df_equals(modin_series.dt.is_quarter_start, pandas_series.dt.is_quarter_start)
    df_equals(modin_series.dt.is_quarter_end, pandas_series.dt.is_quarter_end)
    df_equals(modin_series.dt.is_year_start, pandas_series.dt.is_year_start)
    df_equals(modin_series.dt.is_year_end, pandas_series.dt.is_year_end)
    df_equals(modin_series.dt.is_leap_year, pandas_series.dt.is_leap_year)
    df_equals(modin_series.dt.daysinmonth, pandas_series.dt.daysinmonth)
    df_equals(modin_series.dt.days_in_month, pandas_series.dt.days_in_month)
    assert modin_series.dt.tz == pandas_series.dt.tz
    assert modin_series.dt.freq == pandas_series.dt.freq
    df_equals(modin_series.dt.to_period('W'), pandas_series.dt.to_period('W'))
    assert_array_equal(modin_series.dt.to_pydatetime(), pandas_series.dt.to_pydatetime())
    df_equals(modin_series.dt.tz_localize(None), pandas_series.dt.tz_localize(None))
    if timezone:
        df_equals(modin_series.dt.tz_convert(tz='Europe/Berlin'), pandas_series.dt.tz_convert(tz='Europe/Berlin'))
    df_equals(modin_series.dt.normalize(), pandas_series.dt.normalize())
    df_equals(modin_series.dt.strftime('%B %d, %Y, %r'), pandas_series.dt.strftime('%B %d, %Y, %r'))
    df_equals(modin_series.dt.round('h'), pandas_series.dt.round('h'))
    df_equals(modin_series.dt.floor('h'), pandas_series.dt.floor('h'))
    df_equals(modin_series.dt.ceil('h'), pandas_series.dt.ceil('h'))
    df_equals(modin_series.dt.month_name(), pandas_series.dt.month_name())
    df_equals(modin_series.dt.day_name(), pandas_series.dt.day_name())
    modin_series = pd.Series(pd.to_timedelta(np.arange(128), unit='d'))
    pandas_series = pandas.Series(pandas.to_timedelta(np.arange(128), unit='d'))
    assert_array_equal(modin_series.dt.to_pytimedelta(), pandas_series.dt.to_pytimedelta())
    df_equals(modin_series.dt.total_seconds(), pandas_series.dt.total_seconds())
    df_equals(modin_series.dt.days, pandas_series.dt.days)
    df_equals(modin_series.dt.seconds, pandas_series.dt.seconds)
    df_equals(modin_series.dt.microseconds, pandas_series.dt.microseconds)
    df_equals(modin_series.dt.nanoseconds, pandas_series.dt.nanoseconds)
    df_equals(modin_series.dt.components, pandas_series.dt.components)
    data_per = pd.date_range('1/1/2012', periods=128, freq='M')
    pandas_series = pandas.Series(data_per, index=data_per).dt.to_period()
    modin_series = pd.Series(data_per, index=data_per).dt.to_period()
    df_equals(modin_series.dt.qyear, pandas_series.dt.qyear)
    df_equals(modin_series.dt.start_time, pandas_series.dt.start_time)
    df_equals(modin_series.dt.end_time, pandas_series.dt.end_time)
    df_equals(modin_series.dt.to_timestamp(), pandas_series.dt.to_timestamp())

    def dt_with_empty_partition(lib):
        df = pd.concat([pd.DataFrame([None]), pd.DataFrame([pd.to_timedelta(1)])], axis=1).dropna(axis=1).squeeze(1)
        if isinstance(df, pd.DataFrame) and get_current_execution() != 'BaseOnPython' and (StorageFormat.get() != 'Hdk'):
            assert df._query_compiler._modin_frame._partitions.shape == (1, 2)
        return df.dt.days
    eval_general(pd, pandas, dt_with_empty_partition)
    if timezone is None:
        data = pd.period_range('2016-12-31', periods=128, freq='D')
        modin_series = pd.Series(data)
        pandas_series = pandas.Series(data)
        df_equals(modin_series.dt.asfreq('min'), pandas_series.dt.asfreq('min'))