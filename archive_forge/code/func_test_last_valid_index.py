import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_last_valid_index(data):
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    assert modin_df.last_valid_index() == pandas_df.last_valid_index()