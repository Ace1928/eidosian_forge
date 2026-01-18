import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('q', [0.5, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_groupby_quantile_raises_on_invalid_dtype(q, numeric_only):
    df = DataFrame({'a': [1], 'b': [2.0], 'c': ['x']})
    if numeric_only:
        result = df.groupby('a').quantile(q, numeric_only=numeric_only)
        expected = df.groupby('a')[['b']].quantile(q)
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(TypeError, match="'quantile' cannot be performed against 'object' dtypes!"):
            df.groupby('a').quantile(q, numeric_only=numeric_only)