from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected', [(True, Series([], dtype=np.float64)), (False, Series([], dtype=np.float64))])
def test_mode_empty(self, dropna, expected):
    s = Series([], dtype=np.float64)
    result = s.mode(dropna)
    tm.assert_series_equal(result, expected)