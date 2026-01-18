import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('method', ['cummin', 'cummax'])
def test_cummin_cummax_int_and_float(axis, method):
    data = {'col1': list(range(1000)), 'col2': [i * 0.1 for i in range(1000)]}
    eval_general(*create_test_dfs(data), lambda df: getattr(df, method)(axis=axis))