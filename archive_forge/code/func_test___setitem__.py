import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test___setitem__(data):
    eval_setitem(*create_test_dfs(data), loc=-1, value=1)
    eval_setitem(*create_test_dfs(data), loc=-1, value=lambda df: type(df)(df[df.columns[0]]))
    nrows = len(data[list(data.keys())[0]])
    arr = np.arange(nrows * 2).reshape(-1, 2)
    eval_setitem(*create_test_dfs(data), loc=-1, value=arr)
    eval_setitem(*create_test_dfs(data), col='___NON EXISTENT COLUMN', value=arr.T[0])
    eval_setitem(*create_test_dfs(data), loc=0, value=np.arange(nrows))
    modin_df = pd.DataFrame(columns=data.keys())
    pandas_df = pandas.DataFrame(columns=data.keys())
    for col in modin_df.columns:
        modin_df[col] = np.arange(1000)
    for col in pandas_df.columns:
        pandas_df[col] = np.arange(1000)
    df_equals(modin_df, pandas_df)
    modin_df = pd.DataFrame(columns=modin_df.columns)
    pandas_df = pandas.DataFrame(columns=pandas_df.columns)
    modin_df[modin_df.columns[-1]] = modin_df[modin_df.columns[0]]
    pandas_df[pandas_df.columns[-1]] = pandas_df[pandas_df.columns[0]]
    df_equals(modin_df, pandas_df)
    if not sys.version_info.major == 3 and sys.version_info.minor > 6:
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df['new_col'] = modin_df[[modin_df.columns[0]]].values
        pandas_df['new_col'] = pandas_df[[pandas_df.columns[0]]].values
        df_equals(modin_df, pandas_df)
        assert isinstance(modin_df['new_col'][0], type(pandas_df['new_col'][0]))
    modin_df[1:5] = 10
    pandas_df[1:5] = 10
    df_equals(modin_df, pandas_df)
    modin_df = pd.DataFrame(data).T
    pandas_df = pandas.DataFrame(data).T
    modin_df[modin_df.columns[0]] = 0
    pandas_df[pandas_df.columns[0]] = 0
    df_equals(modin_df, pandas_df)
    modin_df.columns = [str(i) for i in modin_df.columns]
    pandas_df.columns = [str(i) for i in pandas_df.columns]
    modin_df[modin_df.columns[0]] = 0
    pandas_df[pandas_df.columns[0]] = 0
    df_equals(modin_df, pandas_df)
    modin_df[modin_df.columns[0]][modin_df.index[0]] = 12345
    pandas_df[pandas_df.columns[0]][pandas_df.index[0]] = 12345
    df_equals(modin_df, pandas_df)
    modin_df[1:5] = 10
    pandas_df[1:5] = 10
    df_equals(modin_df, pandas_df)