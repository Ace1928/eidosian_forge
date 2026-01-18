import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_max_min_object_multiple_columns(using_array_manager):
    df = DataFrame({'A': [1, 1, 2, 2, 3], 'B': [1, 'foo', 2, 'bar', False], 'C': ['a', 'b', 'c', 'd', 'e']})
    df._consolidate_inplace()
    if not using_array_manager:
        assert len(df._mgr.blocks) == 2
    gb = df.groupby('A')
    result = gb[['C']].max()
    ei = Index([1, 2, 3], name='A')
    expected = DataFrame({'C': ['b', 'd', 'e']}, index=ei)
    tm.assert_frame_equal(result, expected)
    result = gb[['C']].min()
    ei = Index([1, 2, 3], name='A')
    expected = DataFrame({'C': ['a', 'c', 'e']}, index=ei)
    tm.assert_frame_equal(result, expected)