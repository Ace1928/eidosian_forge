from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('value, string_value', [(None, 'None'), (np.nan, 'nan'), (NA, '<NA>')])
def test_astype_to_str_preserves_na(self, value, string_value):
    ser = Series(['a', 'b', value], dtype=object)
    result = ser.astype(str)
    expected = Series(['a', 'b', string_value], dtype=object)
    tm.assert_series_equal(result, expected)