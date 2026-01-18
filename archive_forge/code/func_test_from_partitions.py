import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
@pytest.mark.parametrize('column_widths', [None, 'column_widths'])
@pytest.mark.parametrize('row_lengths', [None, 'row_lengths'])
@pytest.mark.parametrize('columns', [None, 'columns'])
@pytest.mark.parametrize('index', [None, 'index'])
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_from_partitions(axis, index, columns, row_lengths, column_widths):
    data = test_data['int_data']
    df1, df2 = (pandas.DataFrame(data), pandas.DataFrame(data))
    num_rows, num_cols = df1.shape
    expected_df = pandas.concat([df1, df2], axis=1 if axis is None else axis)
    index = expected_df.index if index == 'index' else None
    columns = expected_df.columns if columns == 'columns' else None
    row_lengths = None if row_lengths is None else [num_rows, num_rows] if axis == 0 else [num_rows]
    column_widths = None if column_widths is None else [num_cols] if axis == 0 else [num_cols, num_cols]
    futures = []
    if axis is None:
        futures = [[put_func(df1), put_func(df2)]]
    else:
        futures = [put_func(df1), put_func(df2)]
    actual_df = from_partitions(futures, axis, index=index, columns=columns, row_lengths=row_lengths, column_widths=column_widths)
    df_equals(expected_df, actual_df)