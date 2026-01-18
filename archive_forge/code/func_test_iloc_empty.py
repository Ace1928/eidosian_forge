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
def test_iloc_empty():
    pandas_df = pandas.DataFrame(index=range(5))
    modin_df = pd.DataFrame(index=range(5))
    df_equals(pandas_df.iloc[1], modin_df.iloc[1])
    pandas_df.iloc[1] = 3
    modin_df.iloc[1] = 3
    df_equals(pandas_df, modin_df)