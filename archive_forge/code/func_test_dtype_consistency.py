import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
def test_dtype_consistency(dtype):
    res_dtype = pd.DataFrame([1, 2, 3, 4], dtype=dtype).sum().dtype
    assert res_dtype == pandas.api.types.pandas_dtype(dtype)