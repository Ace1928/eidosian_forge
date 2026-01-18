from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_categorical_values_mismatched_dtype(self):
    dti = date_range('2016-01-01', periods=3)
    cat = Categorical(dti)
    result = Index(cat, dti.dtype)
    tm.assert_index_equal(result, dti)
    dti2 = dti.tz_localize('Asia/Tokyo')
    cat2 = Categorical(dti2)
    result = Index(cat2, dti2.dtype)
    tm.assert_index_equal(result, dti2)
    ii = IntervalIndex.from_breaks(range(5))
    cat3 = Categorical(ii)
    result = Index(cat3, dtype=ii.dtype)
    tm.assert_index_equal(result, ii)