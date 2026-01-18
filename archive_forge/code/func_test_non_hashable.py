import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors,expected', [('raise', 'Invalid object type at position 0'), ('ignore', Series([[10.0, 2], 1.0, 'apple'])), ('coerce', Series([np.nan, 1.0, np.nan]))])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_non_hashable(errors, expected):
    ser = Series([[10.0, 2], 1.0, 'apple'])
    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)