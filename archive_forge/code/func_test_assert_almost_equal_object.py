import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_object():
    a = [Timestamp('2011-01-01'), Timestamp('2011-01-01')]
    b = [Timestamp('2011-01-01'), Timestamp('2011-01-01')]
    _assert_almost_equal_both(a, b)