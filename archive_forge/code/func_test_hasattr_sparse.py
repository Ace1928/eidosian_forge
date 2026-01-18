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
@pytest.mark.parametrize('is_sparse_data', [True, False], ids=['is_sparse', 'is_not_sparse'])
def test_hasattr_sparse(is_sparse_data):
    modin_df, pandas_df = create_test_dfs(pandas.arrays.SparseArray(test_data['float_nan_data'].values())) if is_sparse_data else create_test_dfs(test_data['float_nan_data'])
    eval_general(modin_df, pandas_df, lambda df: hasattr(df, 'sparse'))