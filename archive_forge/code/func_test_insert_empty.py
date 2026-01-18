from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_insert_empty(self):
    idx = timedelta_range('1 Day', periods=3)
    td = idx[0]
    result = idx[:0].insert(0, td)
    assert result.freq == 'D'
    with pytest.raises(IndexError, match='loc must be an integer between'):
        result = idx[:0].insert(1, td)
    with pytest.raises(IndexError, match='loc must be an integer between'):
        result = idx[:0].insert(-1, td)