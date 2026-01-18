import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_datetime_index():
    index = pd.date_range('2001-01-01', periods=100)
    mi = MultiIndex.from_arrays([index])
    assert mi.get_loc('2001-01') == slice(0, 31, None)
    assert index.get_loc('2001-01') == slice(0, 31, None)
    loc = mi[::2].get_loc('2001-01')
    expected = index[::2].get_loc('2001-01')
    assert loc == expected
    loc = mi.repeat(2).get_loc('2001-01')
    expected = index.repeat(2).get_loc('2001-01')
    assert loc == expected
    loc = mi.append(mi).get_loc('2001-01')
    expected = index.append(index).get_loc('2001-01')
    tm.assert_numpy_array_equal(loc.nonzero()[0], expected)