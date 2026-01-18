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
@pytest.mark.parametrize('dtype_backend', ['numpy_nullable', 'pyarrow'])
def test_convert_dtypes_dtype_backend(dtype_backend):
    data = {'a': pd.Series([1, 2, 3], dtype=np.dtype('int32')), 'b': pd.Series(['x', 'y', 'z'], dtype=np.dtype('O')), 'c': pd.Series([True, False, np.nan], dtype=np.dtype('O')), 'd': pd.Series(['h', 'i', np.nan], dtype=np.dtype('O')), 'e': pd.Series([10, np.nan, 20], dtype=np.dtype('float')), 'f': pd.Series([np.nan, 100.5, 200], dtype=np.dtype('float'))}

    def comparator(df1, df2):
        df_equals(df1, df2)
        df_equals(df1.dtypes, df2.dtypes)
    eval_general(*create_test_dfs(data), lambda df: df.convert_dtypes(dtype_backend=dtype_backend), comparator=comparator)