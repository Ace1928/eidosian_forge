import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('min_periods', [1, 3, 5, 6])
def test_corr_min_periods(self, min_periods):
    eval_general(*create_test_dfs({'a': [1, 2, 3], 'b': [3, 1, 5]}), lambda df: df.corr(min_periods=min_periods))
    eval_general(*create_test_dfs({'a': [1, 2, 3, 4, 5, np.nan], 'b': [1, 2, 1, 4, 5, np.nan]}), lambda df: df.corr(min_periods=min_periods))
    eval_general(*create_test_dfs({'a': [1, np.nan, 3, 4, 5, 6], 'b': [1, 2, 1, 4, 5, np.nan]}), lambda df: df.corr(min_periods=min_periods))
    if StorageFormat.get() == 'Pandas':
        modin_df, pandas_df = create_test_dfs({'a': [1, np.nan, 3, 4, 5, 6], 'b': [1, 2, 1, 4, 5, np.nan]})
        modin_df = pd.concat([modin_df.iloc[:3], modin_df.iloc[3:]])
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
        eval_general(modin_df, pandas_df, lambda df: df.corr(min_periods=min_periods))