import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_asfreq_raises(self):
    ser = Series(['a', None, 'b'], dtype=object)
    msg2 = 'Series.interpolate with object dtype'
    msg = 'Invalid fill method'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            ser.interpolate(method='asfreq')