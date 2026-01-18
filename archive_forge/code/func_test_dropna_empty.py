import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dropna_empty(self):
    ser = Series([], dtype=object)
    assert len(ser.dropna()) == 0
    return_value = ser.dropna(inplace=True)
    assert return_value is None
    assert len(ser) == 0
    msg = 'No axis named 1 for object type Series'
    with pytest.raises(ValueError, match=msg):
        ser.dropna(axis=1)