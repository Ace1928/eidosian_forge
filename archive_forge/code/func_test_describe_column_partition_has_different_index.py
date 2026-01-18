import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
def test_describe_column_partition_has_different_index():
    pandas_df = pandas.DataFrame(test_data['int_data'])
    pandas_df['string_column'] = 'abc'
    modin_df = pd.DataFrame(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.describe(include='all'))