import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
@pytest.mark.parametrize('axis', [lib.no_default, 'columns'])
@pytest.mark.parametrize('on', [None, 'DateCol'])
@pytest.mark.parametrize('closed', ['both', 'right'])
@pytest.mark.parametrize('window', [3, '3s'])
def test_dataframe_dt_index(axis, on, closed, window):
    index = pandas.date_range('31/12/2000', periods=12, freq='min')
    data = {'A': range(12), 'B': range(12)}
    pandas_df = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)
    if on is not None and axis == lib.no_default and isinstance(window, str):
        pandas_df[on] = pandas.date_range('22/06/1941', periods=12, freq='min')
        modin_df[on] = pd.date_range('22/06/1941', periods=12, freq='min')
    else:
        on = None
    if axis == 'columns':
        pandas_df = pandas_df.T
        modin_df = modin_df.T
    pandas_rolled = pandas_df.rolling(window=window, on=on, axis=axis, closed=closed)
    modin_rolled = modin_df.rolling(window=window, on=on, axis=axis, closed=closed)
    if isinstance(window, int):
        df_equals(modin_rolled.corr(modin_df, True), pandas_rolled.corr(pandas_df, True))
        df_equals(modin_rolled.corr(modin_df, False), pandas_rolled.corr(pandas_df, False))
        df_equals(modin_rolled.cov(modin_df, True), pandas_rolled.cov(pandas_df, True))
        df_equals(modin_rolled.cov(modin_df, False), pandas_rolled.cov(pandas_df, False))
        if axis == lib.no_default:
            df_equals(modin_rolled.cov(modin_df[modin_df.columns[0]], True), pandas_rolled.cov(pandas_df[pandas_df.columns[0]], True))
            df_equals(modin_rolled.corr(modin_df[modin_df.columns[0]], True), pandas_rolled.corr(pandas_df[pandas_df.columns[0]], True))
    else:
        df_equals(modin_rolled.count(), pandas_rolled.count())
        df_equals(modin_rolled.skew(), pandas_rolled.skew())
        df_equals(modin_rolled.apply(np.sum, raw=True), pandas_rolled.apply(np.sum, raw=True))
        df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
        df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))