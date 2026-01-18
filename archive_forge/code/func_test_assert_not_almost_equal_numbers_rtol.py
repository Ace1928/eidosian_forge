import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1e-05, 1.2e-05), (1e-06, 5e-06)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    _assert_not_almost_equal_both(a, b, rtol=0.05)