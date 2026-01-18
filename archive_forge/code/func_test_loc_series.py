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
def test_loc_series():
    md_df, pd_df = create_test_dfs({'a': [1, 2], 'b': [3, 4]})
    pd_df.loc[pd_df['a'] > 1, 'b'] = np.log(pd_df['b'])
    md_df.loc[md_df['a'] > 1, 'b'] = np.log(md_df['b'])
    df_equals(pd_df, md_df)