from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_listlike_invalid(self):
    ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
    msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
    with pytest.raises(TypeError, match=msg):
        ser.fillna([1, 2])
    msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
    with pytest.raises(TypeError, match=msg):
        ser.fillna((1, 2))