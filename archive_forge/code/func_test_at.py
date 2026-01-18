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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_at(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    key1 = modin_df.columns[0]
    df_equals(modin_df.at[0, key1], pandas_df.at[0, key1])
    df_equals(modin_df.loc[0].at[key1], pandas_df.loc[0].at[key1])
    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()
    modin_df_copy.at[1, key1] = modin_df.at[0, key1]
    pandas_df_copy.at[1, key1] = pandas_df.at[0, key1]
    df_equals(modin_df_copy, pandas_df_copy)