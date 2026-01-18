from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('array,expected,dtype', [([0, 1j, 1, 1 + 1j, 1 + 2j], Series([0j, 1j, 1 + 0j, 1 + 1j, 1 + 2j], dtype=np.complex128), np.complex128), ([1 + 1j, 2j, 1 + 1j, 2j, 3], Series([2j, 1 + 1j], dtype=np.complex64), np.complex64)])
def test_multimode_complex(self, array, expected, dtype):
    result = Series(array, dtype=dtype).mode()
    tm.assert_series_equal(result, expected)