import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
def test_merge_asof_merge_options():
    modin_quotes = pd.DataFrame({'time': [pd.Timestamp('2016-05-25 13:30:00.023'), pd.Timestamp('2016-05-25 13:30:00.023'), pd.Timestamp('2016-05-25 13:30:00.030'), pd.Timestamp('2016-05-25 13:30:00.041'), pd.Timestamp('2016-05-25 13:30:00.048'), pd.Timestamp('2016-05-25 13:30:00.049'), pd.Timestamp('2016-05-25 13:30:00.072'), pd.Timestamp('2016-05-25 13:30:00.075')], 'ticker': ['GOOG', 'MSFT', 'MSFT', 'MSFT', 'GOOG', 'AAPL', 'GOOG', 'MSFT'], 'bid': [720.5, 51.95, 51.97, 51.99, 720.5, 97.99, 720.5, 52.01], 'ask': [720.93, 51.96, 51.98, 52.0, 720.93, 98.01, 720.88, 52.03]})
    modin_trades = pd.DataFrame({'time': [pd.Timestamp('2016-05-25 13:30:00.023'), pd.Timestamp('2016-05-25 13:30:00.038'), pd.Timestamp('2016-05-25 13:30:00.048'), pd.Timestamp('2016-05-25 13:30:00.048'), pd.Timestamp('2016-05-25 13:30:00.048')], 'ticker2': ['MSFT', 'MSFT', 'GOOG', 'GOOG', 'AAPL'], 'price': [51.95, 51.95, 720.77, 720.92, 98.0], 'quantity': [75, 155, 100, 100, 100]})
    pandas_quotes, pandas_trades = (to_pandas(modin_quotes), to_pandas(modin_trades))
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(modin_quotes, modin_trades, on='time', left_by='ticker', right_by='ticker2')
    df_equals(pandas.merge_asof(pandas_quotes, pandas_trades, on='time', left_by='ticker', right_by='ticker2'), modin_result)
    pandas_trades['ticker'] = pandas_trades['ticker2']
    modin_trades['ticker'] = modin_trades['ticker2']
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(modin_quotes, modin_trades, on='time', by='ticker')
    df_equals(pandas.merge_asof(pandas_quotes, pandas_trades, on='time', by='ticker'), modin_result)
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(modin_quotes, modin_trades, on='time', by='ticker', tolerance=pd.Timedelta('2ms'))
    df_equals(pandas.merge_asof(pandas_quotes, pandas_trades, on='time', by='ticker', tolerance=pd.Timedelta('2ms')), modin_result)
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(modin_quotes, modin_trades, on='time', by='ticker', direction='forward')
    df_equals(pandas.merge_asof(pandas_quotes, pandas_trades, on='time', by='ticker', direction='forward'), modin_result)
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(modin_quotes, modin_trades, on='time', by='ticker', tolerance=pd.Timedelta('10ms'), allow_exact_matches=False)
    df_equals(pandas.merge_asof(pandas_quotes, pandas_trades, on='time', by='ticker', tolerance=pd.Timedelta('10ms'), allow_exact_matches=False), modin_result)