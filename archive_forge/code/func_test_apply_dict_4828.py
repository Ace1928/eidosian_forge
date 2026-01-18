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
def test_apply_dict_4828():
    data = [[2, 4], [1, 3]]
    modin_df1, pandas_df1 = create_test_dfs(data)
    eval_general(modin_df1, pandas_df1, lambda df: df.apply({0: lambda x: x ** 2}))
    eval_general(modin_df1, pandas_df1, lambda df: df.apply({0: lambda x: x ** 2}, axis=1))
    modin_df2, pandas_df2 = create_test_dfs(data, index=[2, 3])
    modin_df3 = pd.concat([modin_df1, modin_df2], axis=0)
    pandas_df3 = pandas.concat([pandas_df1, pandas_df2], axis=0)
    eval_general(modin_df3, pandas_df3, lambda df: df.apply({0: lambda x: x ** 2}))
    eval_general(modin_df3, pandas_df3, lambda df: df.apply({0: lambda x: x ** 2}, axis=1))
    modin_df4, pandas_df4 = create_test_dfs(data, columns=[2, 3])
    modin_df5 = pd.concat([modin_df1, modin_df4], axis=1)
    pandas_df5 = pandas.concat([pandas_df1, pandas_df4], axis=1)
    eval_general(modin_df5, pandas_df5, lambda df: df.apply({0: lambda x: x ** 2}))
    eval_general(modin_df5, pandas_df5, lambda df: df.apply({0: lambda x: x ** 2}, axis=1))