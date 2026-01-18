import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, limit_direction, expected', [('pad', 'backward', 'forward'), ('ffill', 'backward', 'forward'), ('backfill', 'forward', 'backward'), ('bfill', 'forward', 'backward'), ('pad', 'both', 'forward'), ('ffill', 'both', 'forward'), ('backfill', 'both', 'backward'), ('bfill', 'both', 'backward')])
def test_interp_limit_direction_raises(self, method, limit_direction, expected):
    s = Series([1, 2, 3])
    msg = f"`limit_direction` must be '{expected}' for method `{method}`"
    msg2 = 'Series.interpolate with method='
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            s.interpolate(method=method, limit_direction=limit_direction)