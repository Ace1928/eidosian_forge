from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_ea_values_mismatched_categorical_dtype(self):
    dti = date_range('2016-01-01', periods=3)
    result = Index(dti, dtype='category')
    expected = CategoricalIndex(dti)
    tm.assert_index_equal(result, expected)
    dti2 = date_range('2016-01-01', periods=3, tz='US/Pacific')
    result = Index(dti2, dtype='category')
    expected = CategoricalIndex(dti2)
    tm.assert_index_equal(result, expected)