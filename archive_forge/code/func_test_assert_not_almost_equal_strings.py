import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [('abc', 'abcd'), ('abc', 'abd'), ('abc', 1), ('abc', [1])])
def test_assert_not_almost_equal_strings(a, b):
    _assert_not_almost_equal_both(a, b)