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
@pytest.mark.parametrize('errors', ['raise', 'ignore'])
def test_astype_errors(errors):
    data = {'a': ['a', 2, -1]}
    modin_df, pandas_df = create_test_dfs(data)
    expected_exception = None
    if errors == 'raise':
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7025')
    eval_general(modin_df, pandas_df, lambda df: df.astype('int', errors=errors), comparator_kwargs={'check_dtypes': errors != 'ignore'}, expected_exception=expected_exception)