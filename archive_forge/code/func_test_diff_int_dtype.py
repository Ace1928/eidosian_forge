import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_int_dtype(self):
    a = 10000000000000000
    b = a + 1
    ser = Series([a, b])
    rs = DataFrame({'s': ser}).diff()
    assert rs.s[1] == 1