import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
@pytest.mark.parametrize('a,b', NESTED_CASES)
def test_assert_almost_equal_array_nested(a, b):
    _assert_almost_equal_both(a, b)