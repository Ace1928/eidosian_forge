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
@pytest.mark.parametrize('reverse_order', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_loc_order(loc_iter_dfs, reverse_order, axis):
    md_df, pd_df = loc_iter_dfs
    select = [slice(None), slice(None)]
    select[axis] = sorted(pd_df.axes[axis][:-1], reverse=reverse_order)
    select = tuple(select)
    df_equals(pd_df.loc[select], md_df.loc[select])