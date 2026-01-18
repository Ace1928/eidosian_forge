import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_invalid_float_limit(self, nontemporal_method):
    s = Series([1, 2, np.nan, 4])
    method, kwargs = nontemporal_method
    limit = 2.0
    with pytest.raises(ValueError, match='Limit must be an integer'):
        s.interpolate(limit=limit, method=method, **kwargs)