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
def test_loc_6774():
    modin_df, pandas_df = create_test_dfs({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
    pandas_df.loc[:, 'c'] = [10, 20, 30, 40, 51]
    modin_df.loc[:, 'c'] = [10, 20, 30, 40, 51]
    df_equals(modin_df, pandas_df)
    pandas_df.loc[2:, 'y'] = [30, 40, 51]
    modin_df.loc[2:, 'y'] = [30, 40, 51]
    df_equals(modin_df, pandas_df)
    pandas_df.loc[:, ['b', 'c', 'd']] = pd.DataFrame([[10, 20, 30, 40, 50], [10, 20, 30, 40], [10, 20, 30]]).transpose().values
    modin_df.loc[:, ['b', 'c', 'd']] = pd.DataFrame([[10, 20, 30, 40, 50], [10, 20, 30, 40], [10, 20, 30]]).transpose().values
    df_equals(modin_df, pandas_df)