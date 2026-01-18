import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_not_equal_sets():
    msg = '{1, 2, 3} != {1, 2, 4}'
    with pytest.raises(AssertionError, match=msg):
        _assert_almost_equal_both({1, 2, 3}, {1, 2, 4})