from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_getitem_mixed_index_no_fallback(self):
    ser = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
    with pytest.raises(KeyError, match='^0$'):
        ser.at[0]
    with pytest.raises(KeyError, match='^4$'):
        ser.at[4]