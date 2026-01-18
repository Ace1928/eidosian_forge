import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', [np.arange(1, 10000, dtype=np.float32), [pd.Series([1, 2, 3], dtype='int32'), pandas.Series([4, 5, 6], dtype='int64'), np.array([7, 8, 9], dtype=np.float32)], pandas.Categorical([1, 2, 3, 4, 5])])
def test_constructor_dtypes(data):
    modin_df, pandas_df = create_test_dfs(data)
    df_equals(modin_df, pandas_df)