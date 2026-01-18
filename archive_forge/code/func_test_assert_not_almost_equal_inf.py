import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_not_almost_equal_inf():
    _assert_not_almost_equal_both(np.inf, 0)