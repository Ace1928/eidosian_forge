import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_extended_gcd(self, simple_index):
    index = simple_index
    result = index._extended_gcd(6, 10)
    assert result[0] == result[1] * 6 + result[2] * 10
    assert 2 == result[0]
    result = index._extended_gcd(10, 6)
    assert 2 == result[1] * 10 + result[2] * 6
    assert 2 == result[0]