import pytest
from pandas import (
import pandas._testing as tm
def test_delitem_object_index(self, using_infer_string):
    dtype = 'string[pyarrow_numpy]' if using_infer_string else object
    s = Series(1, index=Index(['a'], dtype=dtype))
    del s['a']
    tm.assert_series_equal(s, Series(dtype='int64', index=Index([], dtype=dtype)))
    s['a'] = 1
    tm.assert_series_equal(s, Series(1, index=Index(['a'], dtype=dtype)))
    del s['a']
    tm.assert_series_equal(s, Series(dtype='int64', index=Index([], dtype=dtype)))