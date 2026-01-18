from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_center_ljust_rjust_mixed_object():
    s = Series(['a', np.nan, 'b', True, datetime.today(), 'c', 'eee', None, 1, 2.0])
    result = s.str.center(5)
    expected = Series(['  a  ', np.nan, '  b  ', np.nan, np.nan, '  c  ', ' eee ', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.ljust(5)
    expected = Series(['a    ', np.nan, 'b    ', np.nan, np.nan, 'c    ', 'eee  ', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.rjust(5)
    expected = Series(['    a', np.nan, '    b', np.nan, np.nan, '    c', '  eee', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)