import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('subset_operand', ['left', 'right'])
def test_mismatched_col_partitions(subset_operand):
    data = [0, 1, 2, 3]
    modin_df1, pandas_df1 = create_test_dfs({'a': data, 'b': data})
    modin_df_tmp, pandas_df_tmp = create_test_dfs({'c': data})
    modin_df2 = pd.concat([modin_df1, modin_df_tmp], axis=1)
    pandas_df2 = pandas.concat([pandas_df1, pandas_df_tmp], axis=1)
    if subset_operand == 'right':
        modin_res = modin_df2 + modin_df1
        pandas_res = pandas_df2 + pandas_df1
    else:
        modin_res = modin_df1 + modin_df2
        pandas_res = pandas_df1 + pandas_df2
    df_equals(modin_res, pandas_res)