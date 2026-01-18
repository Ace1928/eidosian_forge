import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('method', ['median', 'skew'])
def test_median_skew(axis, skipna, method):
    eval_general(*create_test_dfs(test_data['float_nan_data']), lambda df: getattr(df, method)(axis=axis, skipna=skipna))