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
def test_insert_modin_array():
    from modin.numpy import array
    data = {'col1': [1, 2, 3], 'col2': [2, 3, 4]}
    modin_df1, modin_df2 = (pd.DataFrame(data), pd.DataFrame(data))
    np_value = np.array([7, 7, 7])
    md_np_value = array(np_value)
    modin_df1.insert(1, 'new_col', np_value)
    modin_df2.insert(1, 'new_col', md_np_value)
    df_equals(modin_df1, modin_df2)