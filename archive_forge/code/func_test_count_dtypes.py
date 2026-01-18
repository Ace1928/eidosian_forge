import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.skipif(StorageFormat.get() == 'Hdk', reason='https://github.com/intel-ai/hdk/issues/513')
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_count_dtypes(data):
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    eval_general(modin_df, pandas_df, lambda df: df.isna().count(axis=0))