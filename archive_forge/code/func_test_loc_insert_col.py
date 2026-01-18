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
@pytest.mark.parametrize('columns', [10, (100, 102), (2, 6), [10, 11, 12], 'a', ['b', 'c', 'd']])
def test_loc_insert_col(columns):
    pandas_df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])
    modin_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    if isinstance(columns, tuple) and len(columns) == 2:

        def _test_loc_cols(df):
            df.loc[:, columns[0]:columns[1]] = 1
    else:

        def _test_loc_cols(df):
            df.loc[:, columns] = 1
    eval_general(modin_df, pandas_df, _test_loc_cols)