import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('method', ['median', 'skew'])
def test_median_skew_transposed(axis, method):
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: getattr(df.T, method)(axis=axis))