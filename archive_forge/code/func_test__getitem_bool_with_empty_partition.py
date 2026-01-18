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
def test__getitem_bool_with_empty_partition():
    size = MinPartitionSize.get()
    pandas_series = pandas.Series([True if i % 2 else False for i in range(size)])
    modin_series = pd.Series(pandas_series)
    pandas_df = pandas.DataFrame([i for i in range(size + 1)])
    pandas_df.iloc[size] = np.nan
    modin_df = pd.DataFrame(pandas_df)
    pandas_tmp_result = pandas_df.dropna()
    modin_tmp_result = modin_df.dropna()
    eval_general(modin_tmp_result, pandas_tmp_result, lambda df: df[modin_series] if isinstance(df, pd.DataFrame) else df[pandas_series])