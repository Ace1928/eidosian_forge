import os
import subprocess
import sys
import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
import pandas._testing as tm
from pandas.core.internals import (
def test_dataframe_creation():
    msg = 'data_manager option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.data_manager', 'block'):
            df_block = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': [4, 5, 6]})
    assert isinstance(df_block._mgr, BlockManager)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.data_manager', 'array'):
            df_array = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': [4, 5, 6]})
    assert isinstance(df_array._mgr, ArrayManager)
    tm.assert_frame_equal(df_block, df_array)
    result = df_block._as_manager('block')
    assert isinstance(result._mgr, BlockManager)
    result = df_block._as_manager('array')
    assert isinstance(result._mgr, ArrayManager)
    tm.assert_frame_equal(result, df_block)
    assert all((array_equivalent(left, right) for left, right in zip(result._mgr.arrays, df_array._mgr.arrays)))
    result = df_array._as_manager('array')
    assert isinstance(result._mgr, ArrayManager)
    result = df_array._as_manager('block')
    assert isinstance(result._mgr, BlockManager)
    tm.assert_frame_equal(result, df_array)
    assert len(result._mgr.blocks) == 2