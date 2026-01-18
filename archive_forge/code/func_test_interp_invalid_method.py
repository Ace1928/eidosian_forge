import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('invalid_method', [None, 'nonexistent_method'])
def test_interp_invalid_method(self, invalid_method):
    s = Series([1, 3, np.nan, 12, np.nan, 25])
    msg = f"method must be one of.* Got '{invalid_method}' instead"
    if invalid_method is None:
        msg = "'method' should be a string, not None"
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method=invalid_method)
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method=invalid_method, limit=-1)