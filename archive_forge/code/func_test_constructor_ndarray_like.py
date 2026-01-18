from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('array', [np.arange(5), np.array(['a', 'b', 'c']), date_range('2000-01-01', periods=3).values])
def test_constructor_ndarray_like(self, array):

    class ArrayLike:

        def __init__(self, array) -> None:
            self.array = array

        def __array__(self, dtype=None) -> np.ndarray:
            return self.array
    expected = Index(array)
    result = Index(ArrayLike(array))
    tm.assert_index_equal(result, expected)