import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def test_groupby_ohlc():
    pandas_df = pandas.DataFrame(np.random.randint(0, 100, (50, 2)), columns=['stock A', 'stock B'])
    pandas_df['Date'] = pandas.concat([pandas.date_range('1/1/2000', periods=10, freq='min').to_series()] * 5).reset_index(drop=True)
    modin_df = pd.DataFrame(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.groupby('Date')['stock A'].ohlc())
    pandas_multiindex_result = pandas_df.groupby('Date')[['stock A']].ohlc()
    with warns_that_defaulting_to_pandas():
        modin_multiindex_result = modin_df.groupby('Date')[['stock A']].ohlc()
    df_equals(modin_multiindex_result, pandas_multiindex_result)
    pandas_multiindex_result = pandas_df.groupby('Date')[['stock A', 'stock B']].ohlc()
    with warns_that_defaulting_to_pandas():
        modin_multiindex_result = modin_df.groupby('Date')[['stock A', 'stock B']].ohlc()
    df_equals(modin_multiindex_result, pandas_multiindex_result)