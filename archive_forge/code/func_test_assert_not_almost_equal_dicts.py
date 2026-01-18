import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [({'a': 1, 'b': 2}, {'a': 1, 'b': 3}), ({'a': 1, 'b': 2}, {'a': 1, 'b': 2, 'c': 3}), ({'a': 1}, 1), ({'a': 1}, 'abc'), ({'a': 1}, [1])])
def test_assert_not_almost_equal_dicts(a, b):
    _assert_not_almost_equal_both(a, b)