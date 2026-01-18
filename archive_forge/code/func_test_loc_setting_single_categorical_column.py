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
def test_loc_setting_single_categorical_column():
    modin_df = pd.DataFrame({'status': ['a', 'b', 'c']}, dtype='category')
    pandas_df = pandas.DataFrame({'status': ['a', 'b', 'c']}, dtype='category')
    modin_df.loc[1:3, 'status'] = 'a'
    pandas_df.loc[1:3, 'status'] = 'a'
    df_equals(modin_df, pandas_df)