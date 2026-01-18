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
def test_lazy_eval_index():
    modin_df, pandas_df = create_test_dfs({'col0': [0, 1]})

    def func(df):
        df_copy = df[df['col0'] < 6].copy()
        df_copy['col0'] = df_copy['col0'].apply(lambda x: x + 1)
        return df_copy
    eval_general(modin_df, pandas_df, func)