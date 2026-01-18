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
@pytest.mark.parametrize('limit_area', [None, 'inside', 'outside'])
@pytest.mark.parametrize('method', ['ffill', 'bfill'])
def test_ffill_bfill_limit_area(method, limit_area):
    modin_df, pandas_df = create_test_dfs([1, None, 2, None])
    eval_general(modin_df, pandas_df, lambda df: getattr(df, method)(limit_area=limit_area))