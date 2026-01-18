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
def test_dropna_subset(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    if 'empty_data' not in request.node.name:
        column_subset = modin_df.columns[0:2]
        df_equals(modin_df.dropna(how='all', subset=column_subset), pandas_df.dropna(how='all', subset=column_subset))
        df_equals(modin_df.dropna(how='any', subset=column_subset), pandas_df.dropna(how='any', subset=column_subset))
        row_subset = modin_df.index[0:2]
        df_equals(modin_df.dropna(how='all', axis=1, subset=row_subset), pandas_df.dropna(how='all', axis=1, subset=row_subset))
        df_equals(modin_df.dropna(how='any', axis=1, subset=row_subset), pandas_df.dropna(how='any', axis=1, subset=row_subset))