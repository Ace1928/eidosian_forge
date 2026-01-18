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
@pytest.mark.parametrize('name', [None, 'foo'])
@pytest.mark.parametrize('dtype_ordered', [True, False])
@pytest.mark.parametrize('series_ordered', [True, False])
def test_astype_categorical_to_categorical(self, name, dtype_ordered, series_ordered):
    s_data = list('abcaacbab')
    s_dtype = CategoricalDtype(list('bac'), ordered=series_ordered)
    ser = Series(s_data, dtype=s_dtype, name=name)
    dtype = CategoricalDtype(ordered=dtype_ordered)
    result = ser.astype(dtype)
    exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
    expected = Series(s_data, name=name, dtype=exp_dtype)
    tm.assert_series_equal(result, expected)
    dtype = CategoricalDtype(list('adc'), dtype_ordered)
    result = ser.astype(dtype)
    expected = Series(s_data, name=name, dtype=dtype)
    tm.assert_series_equal(result, expected)
    if dtype_ordered is False:
        expected = ser
        result = ser.astype('category')
        tm.assert_series_equal(result, expected)