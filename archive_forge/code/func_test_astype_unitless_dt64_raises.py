from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_unitless_dt64_raises(self):
    ser = Series(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]')
    df = ser.to_frame()
    msg = "Casting to unit-less dtype 'datetime64' is not supported"
    with pytest.raises(TypeError, match=msg):
        ser.astype(np.datetime64)
    with pytest.raises(TypeError, match=msg):
        df.astype(np.datetime64)
    with pytest.raises(TypeError, match=msg):
        ser.astype('datetime64')
    with pytest.raises(TypeError, match=msg):
        df.astype('datetime64')