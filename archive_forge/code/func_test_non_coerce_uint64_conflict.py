import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors,exp', [('ignore', Series(['12345678901234567890', '1234567890', 'ITEM'])), ('raise', 'Unable to parse string')])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_non_coerce_uint64_conflict(errors, exp):
    ser = Series(['12345678901234567890', '1234567890', 'ITEM'])
    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, ser)