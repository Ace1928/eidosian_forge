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
def test_drop_api_equivalence():
    frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    modin_df = pd.DataFrame(frame_data, index=['a', 'b', 'c'], columns=['d', 'e', 'f'])
    modin_df1 = modin_df.drop('a')
    modin_df2 = modin_df.drop(index='a')
    df_equals(modin_df1, modin_df2)
    modin_df1 = modin_df.drop('d', axis=1)
    modin_df2 = modin_df.drop(columns='d')
    df_equals(modin_df1, modin_df2)
    modin_df1 = modin_df.drop(labels='e', axis=1)
    modin_df2 = modin_df.drop(columns='e')
    df_equals(modin_df1, modin_df2)
    modin_df1 = modin_df.drop(['a'], axis=0)
    modin_df2 = modin_df.drop(index=['a'])
    df_equals(modin_df1, modin_df2)
    modin_df1 = modin_df.drop(['a'], axis=0).drop(['d'], axis=1)
    modin_df2 = modin_df.drop(index=['a'], columns=['d'])
    df_equals(modin_df1, modin_df2)
    with pytest.raises(ValueError):
        modin_df.drop(labels='a', index='b')
    with pytest.raises(ValueError):
        modin_df.drop(labels='a', columns='b')
    with pytest.raises(ValueError):
        modin_df.drop(axis=1)