from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_nansum_buglet(self):
    ser = Series([1.0, np.nan], index=[0, 1])
    result = np.nansum(ser)
    tm.assert_almost_equal(result, 1)