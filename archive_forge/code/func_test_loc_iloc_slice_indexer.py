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
@pytest.mark.parametrize('locator_name', ['loc', 'iloc'])
@pytest.mark.parametrize('slice_indexer', [slice(None, None, -2), slice(1, 10, None), slice(None, 10, None), slice(10, None, None), slice(10, None, -2), slice(-10, None, -2), slice(None, 1000000000, None)])
def test_loc_iloc_slice_indexer(locator_name, slice_indexer):
    md_df, pd_df = create_test_dfs(test_data_values[0])
    shifted_index = pandas.RangeIndex(1, len(md_df) + 1)
    md_df.index = shifted_index
    pd_df.index = shifted_index
    eval_general(md_df, pd_df, lambda df: getattr(df, locator_name)[slice_indexer])