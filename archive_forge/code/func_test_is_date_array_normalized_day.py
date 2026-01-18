import numpy as np
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas._libs.tslibs.vectorized import is_date_array_normalized
def test_is_date_array_normalized_day(self):
    arr = day_arr
    abbrev = 'D'
    unit = abbrev_to_npy_unit(abbrev)
    result = is_date_array_normalized(arr.view('i8'), None, unit)
    assert result is True