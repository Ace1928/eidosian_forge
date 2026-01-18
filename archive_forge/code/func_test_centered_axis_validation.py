import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_centered_axis_validation(step):
    msg = "The 'axis' keyword in Series.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Series(np.ones(10)).rolling(window=3, center=True, axis=0, step=step).mean()
    msg = 'No axis named 1 for object type Series'
    with pytest.raises(ValueError, match=msg):
        Series(np.ones(10)).rolling(window=3, center=True, axis=1, step=step).mean()
    df = DataFrame(np.ones((10, 10)))
    msg = "The 'axis' keyword in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.rolling(window=3, center=True, axis=0, step=step).mean()
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.rolling(window=3, center=True, axis=1, step=step).mean()
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        df.rolling(window=3, center=True, axis=2, step=step).mean()