import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_2d_ndarray(self, obj, other, expected):
    result = obj.dot(other.values)
    assert np.all(result == expected.values)