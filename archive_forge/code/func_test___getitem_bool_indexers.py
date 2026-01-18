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
def test___getitem_bool_indexers(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    indices = [i % 3 == 0 for i in range(len(modin_df.index))]
    columns = [i % 5 == 0 for i in range(len(modin_df.columns))]
    modin_result = modin_df.loc[indices, columns]
    pandas_result = pandas_df.loc[indices, columns]
    df_equals(modin_result, pandas_result)
    df_equals(modin_df.loc[pd.Series(indices), pd.Series(columns, index=modin_df.columns)], pandas_df.loc[pandas.Series(indices), pandas.Series(columns, index=modin_df.columns)])