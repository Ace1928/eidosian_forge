import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1, 'abc'), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    _assert_not_almost_equal_both(a, b)