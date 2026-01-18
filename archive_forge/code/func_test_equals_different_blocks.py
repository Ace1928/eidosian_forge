import numpy as np
from pandas import (
import pandas._testing as tm
def test_equals_different_blocks(self, using_array_manager, using_infer_string):
    df0 = DataFrame({'A': ['x', 'y'], 'B': [1, 2], 'C': ['w', 'z']})
    df1 = df0.reset_index()[['A', 'B', 'C']]
    if not using_array_manager and (not using_infer_string):
        assert df0._mgr.blocks[0].dtype != df1._mgr.blocks[0].dtype
    tm.assert_frame_equal(df0, df1)
    assert df0.equals(df1)
    assert df1.equals(df0)