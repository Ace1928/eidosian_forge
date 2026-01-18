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
@pytest.mark.parametrize('index', [['row1', 'row2', 'row3']])
@pytest.mark.parametrize('columns', [['col1', 'col2']])
def test_loc_assignment(index, columns):
    md_df, pd_df = create_test_dfs(index=index, columns=columns)
    for i, ind in enumerate(index):
        for j, col in enumerate(columns):
            value_to_assign = int(str(i) + str(j))
            md_df.loc[ind][col] = value_to_assign
            pd_df.loc[ind][col] = value_to_assign
    df_equals(md_df, pd_df)