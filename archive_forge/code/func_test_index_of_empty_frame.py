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
def test_index_of_empty_frame():
    md_df, pd_df = create_test_dfs({}, index=pandas.Index([], name='index name'), columns=['a', 'b'])
    assert md_df.empty and pd_df.empty
    df_equals(md_df.index, pd_df.index)
    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(data, index=pandas.RangeIndex(len(next(iter(data.values()))), name='index name'))
    md_res = md_df.query(f'{md_df.columns[0]} > {RAND_HIGH}')
    pd_res = pd_df.query(f'{pd_df.columns[0]} > {RAND_HIGH}')
    assert md_res.empty and pd_res.empty
    df_equals(md_res.index, pd_res.index)