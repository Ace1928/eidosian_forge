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
def test_iloc_nested_assignment(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    key1 = modin_df.columns[0]
    key2 = modin_df.columns[1]
    modin_df[key1].iloc[0] = 500
    pandas_df[key1].iloc[0] = 500
    df_equals(modin_df, pandas_df)
    modin_df[key2].iloc[0] = None
    pandas_df[key2].iloc[0] = None
    df_equals(modin_df, pandas_df)