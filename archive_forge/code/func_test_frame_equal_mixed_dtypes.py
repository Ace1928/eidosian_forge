import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('indexer', [(0, 1), (1, 0)])
def test_frame_equal_mixed_dtypes(frame_or_series, any_numeric_ea_dtype, indexer):
    dtypes = (any_numeric_ea_dtype, 'int64')
    obj1 = frame_or_series([1, 2], dtype=dtypes[indexer[0]])
    obj2 = frame_or_series([1, 2], dtype=dtypes[indexer[1]])
    tm.assert_equal(obj1, obj2, check_exact=True, check_dtype=False)