import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_callable():
    """
    Test output on a callable
    """
    x = np.linspace(-2.5, 2.5, 6)
    ser = Series(x)
    result = ser.case_when(caselist=[(lambda df: df < 0, lambda df: -df), (lambda df: df >= 0, lambda df: df)])
    expected = np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    tm.assert_series_equal(result, Series(expected))