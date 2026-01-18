import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['2MS', offsets.MonthBegin(2)])
def test_invalid_window_nonfixed(self, freq, regular):
    msg = '\\<2 \\* MonthBegins\\> is a non-fixed frequency'
    with pytest.raises(ValueError, match=msg):
        regular.rolling(window=freq)