import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_apply_modin_func_4635():
    data = [1]
    modin_df, pandas_df = create_test_dfs(data)
    df_equals(modin_df.apply(pd.Series.sum), pandas_df.apply(pandas.Series.sum))
    data = {'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]}
    modin_df, pandas_df = create_test_dfs(data)
    modin_df = modin_df.set_index(['a'])
    pandas_df = pandas_df.set_index(['a'])
    df_equals(modin_df.groupby('a', group_keys=False).apply(pd.DataFrame.sample, n=1), pandas_df.groupby('a', group_keys=False).apply(pandas.DataFrame.sample, n=1))