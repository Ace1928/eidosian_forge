import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_sets():
    _assert_almost_equal_both({1, 2, 3}, {1, 2, 3})