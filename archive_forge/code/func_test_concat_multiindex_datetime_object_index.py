import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiindex_datetime_object_index(self):
    idx = Index([dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)], dtype='object')
    s = Series(['a', 'b'], index=MultiIndex.from_arrays([[1, 2], idx[:-1]], names=['first', 'second']))
    s2 = Series(['a', 'b'], index=MultiIndex.from_arrays([[1, 2], idx[::2]], names=['first', 'second']))
    mi = MultiIndex.from_arrays([[1, 2, 2], idx], names=['first', 'second'])
    assert mi.levels[1].dtype == object
    expected = DataFrame([['a', 'a'], ['b', np.nan], [np.nan, 'b']], index=mi)
    result = concat([s, s2], axis=1)
    tm.assert_frame_equal(result, expected)