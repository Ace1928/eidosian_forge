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
def test_datetime64tz_aware(self, unit):
    dti = Index([Timestamp('20160101', tz='US/Eastern'), Timestamp('20160101', tz='US/Eastern')]).as_unit(unit)
    ser = Series(dti)
    result = ser.unique()
    expected = dti[:1]._data
    tm.assert_extension_array_equal(result, expected)
    result = dti.unique()
    expected = dti[:1]
    tm.assert_index_equal(result, expected)
    result = pd.unique(ser)
    expected = dti[:1]._data
    tm.assert_extension_array_equal(result, expected)
    result = pd.unique(dti)
    expected = dti[:1]
    tm.assert_index_equal(result, expected)