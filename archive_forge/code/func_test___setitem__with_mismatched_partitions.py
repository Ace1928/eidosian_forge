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
def test___setitem__with_mismatched_partitions():
    with ensure_clean('.csv') as fname:
        np.savetxt(fname, np.random.randint(0, 100, size=(200000, 99)), delimiter=',')
        modin_df = pd.read_csv(fname)
        pandas_df = pandas.read_csv(fname)
        modin_df['new'] = pd.Series(list(range(len(modin_df))))
        pandas_df['new'] = pandas.Series(list(range(len(pandas_df))))
        df_equals(modin_df, pandas_df)