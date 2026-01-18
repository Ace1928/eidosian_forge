import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_scipy', [False, pytest.param(True, marks=td.skip_if_no('scipy'))])
def test_interp_multiIndex(self, check_scipy):
    idx = MultiIndex.from_tuples([(0, 'a'), (1, 'b'), (2, 'c')])
    s = Series([1, 2, np.nan], index=idx)
    expected = s.copy()
    expected.loc[2] = 2
    result = s.interpolate()
    tm.assert_series_equal(result, expected)
    msg = 'Only `method=linear` interpolation is supported on MultiIndexes'
    if check_scipy:
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='polynomial', order=1)