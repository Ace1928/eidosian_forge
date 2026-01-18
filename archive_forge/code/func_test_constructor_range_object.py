from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_constructor_range_object(self):
    result = RangeIndex(range(1, 5, 2))
    expected = RangeIndex(1, 5, 2)
    tm.assert_index_equal(result, expected, exact=True)