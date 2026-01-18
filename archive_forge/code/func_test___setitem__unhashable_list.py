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
def test___setitem__unhashable_list():
    cols = ['a', 'b']
    modin_df = pd.DataFrame([[0, 0]], columns=cols)
    modin_df[cols] = modin_df[cols]
    pandas_df = pandas.DataFrame([[0, 0]], columns=cols)
    pandas_df[cols] = pandas_df[cols]
    df_equals(modin_df, pandas_df)