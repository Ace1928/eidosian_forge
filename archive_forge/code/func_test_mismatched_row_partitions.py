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
@pytest.mark.parametrize('is_more_other_partitions', [True, False])
@pytest.mark.parametrize('op_type', ['df_ser', 'df_df', 'ser_ser_same_name', 'ser_ser_different_name'])
@pytest.mark.parametrize('is_idx_aligned', [True, False], ids=['idx_aligned', 'idx_not_aligned'])
def test_mismatched_row_partitions(is_idx_aligned, op_type, is_more_other_partitions):
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, pandas_df1 = create_test_dfs({'a': data, 'b': data})
    modin_df, pandas_df = (modin_df1.loc[:2], pandas_df1.loc[:2])
    modin_df2 = pd.concat((modin_df, modin_df))
    pandas_df2 = pandas.concat((pandas_df, pandas_df))
    if is_more_other_partitions:
        modin_df2, modin_df1 = (modin_df1, modin_df2)
        pandas_df2, pandas_df1 = (pandas_df1, pandas_df2)
    if is_idx_aligned:
        if is_more_other_partitions:
            modin_df1.index = pandas_df1.index = pandas_df2.index
        else:
            modin_df2.index = pandas_df2.index = pandas_df1.index
    if op_type == 'df_ser' and (not is_idx_aligned) and is_more_other_partitions:
        eval_general(modin_df2, pandas_df2, lambda df: df / modin_df1.a if isinstance(df, pd.DataFrame) else df / pandas_df1.a, expected_exception=ValueError('cannot reindex on an axis with duplicate labels'))
        return
    if op_type == 'df_ser':
        modin_res = modin_df2 / modin_df1.a
        pandas_res = pandas_df2 / pandas_df1.a
    elif op_type == 'df_df':
        modin_res = modin_df2 / modin_df1
        pandas_res = pandas_df2 / pandas_df1
    elif op_type == 'ser_ser_same_name':
        modin_res = modin_df2.a / modin_df1.a
        pandas_res = pandas_df2.a / pandas_df1.a
    elif op_type == 'ser_ser_different_name':
        modin_res = modin_df2.a / modin_df1.b
        pandas_res = pandas_df2.a / pandas_df1.b
    else:
        raise Exception(f'op_type: {op_type} not supported in test')
    df_equals(modin_res, pandas_res)