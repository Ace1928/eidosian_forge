import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_join_midx_string():
    midx = MultiIndex.from_arrays([Series(['a', 'a', 'c'], dtype=StringDtype()), Series(['a', 'b', 'c'], dtype=StringDtype())], names=['a', 'b'])
    midx2 = MultiIndex.from_arrays([Series(['a'], dtype=StringDtype()), Series(['c'], dtype=StringDtype())], names=['a', 'c'])
    result = midx.join(midx2, how='inner')
    expected = MultiIndex.from_arrays([Series(['a', 'a'], dtype=StringDtype()), Series(['a', 'b'], dtype=StringDtype()), Series(['c', 'c'], dtype=StringDtype())], names=['a', 'b', 'c'])
    tm.assert_index_equal(result, expected)