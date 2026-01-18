import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_invalid_method_and_value(self):
    ser = Series([1, 3, np.nan, 12, np.nan, 25])
    msg = "'fill_value' is not a valid keyword for Series.interpolate"
    msg2 = 'Series.interpolate with method=pad'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            ser.interpolate(fill_value=3, method='pad')