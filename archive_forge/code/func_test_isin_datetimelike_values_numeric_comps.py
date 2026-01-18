from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('dtype1', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]', 'period[D]'])
@pytest.mark.parametrize('dtype', ['i8', 'f8', 'u8'])
def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
    dta = date_range('2013-01-01', periods=3)._values
    arr = Series(dta.view('i8')).array.view(dtype1)
    comps = arr.view('i8').astype(dtype)
    result = algos.isin(comps, arr)
    expected = np.zeros(comps.shape, dtype=bool)
    tm.assert_numpy_array_equal(result, expected)