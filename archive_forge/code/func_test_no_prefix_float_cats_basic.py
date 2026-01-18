import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_no_prefix_float_cats_basic():
    dummies = DataFrame({1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]})
    expected = DataFrame({'': [1.0, 25.0, 2.5, 5.84]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)