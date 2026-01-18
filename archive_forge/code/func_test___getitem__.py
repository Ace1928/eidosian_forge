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
def test___getitem__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    key = modin_df.columns[0]
    modin_col = modin_df.__getitem__(key)
    assert isinstance(modin_col, pd.Series)
    pd_col = pandas_df[key]
    df_equals(pd_col, modin_col)
    slices = [(None, -1), (-1, None), (1, 2), (1, None), (None, 1), (1, -1), (-3, -1), (1, -1, 2), (-1, 1, -1), (None, None, 2)]
    for slice_param in slices:
        s = slice(*slice_param)
        df_equals(modin_df[s], pandas_df[s])
    df_equals(pd.DataFrame([])[:10], pandas.DataFrame([])[:10])