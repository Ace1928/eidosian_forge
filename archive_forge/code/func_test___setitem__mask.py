import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test___setitem__mask():
    data = test_data['int_data']
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    mean = int((RAND_HIGH + RAND_LOW) / 2)
    pandas_df[pandas_df > mean] = -50
    modin_df[modin_df > mean] = -50
    df_equals(modin_df, pandas_df)
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    array = (pandas_df > mean).to_numpy()
    modin_df[array] = -50
    pandas_df[array] = -50
    df_equals(modin_df, pandas_df)
    with pytest.raises(ValueError):
        array = np.array([[1, 2], [3, 4]])
        modin_df[array] = 20