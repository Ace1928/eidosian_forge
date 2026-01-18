import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1, 1.11), (0.1, 0.101), (1.1e-05, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    _assert_not_almost_equal_both(a, b, atol=0.001)