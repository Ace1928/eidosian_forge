from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('propname', PeriodArray._field_ops)
def test_int_properties(self, arr1d, propname):
    pi = self.index_cls(arr1d)
    arr = arr1d
    result = getattr(arr, propname)
    expected = np.array(getattr(pi, propname))
    tm.assert_numpy_array_equal(result, expected)