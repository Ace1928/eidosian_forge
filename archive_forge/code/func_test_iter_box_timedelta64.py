import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_iter_box_timedelta64(self, unit):
    vals = [Timedelta('1 days'), Timedelta('2 days')]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f'timedelta64[{unit}]'
    for res, exp in zip(ser, vals):
        assert isinstance(res, Timedelta)
        assert res == exp
        assert res.unit == unit