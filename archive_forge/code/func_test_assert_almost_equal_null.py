import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_null():
    _assert_almost_equal_both(None, None)