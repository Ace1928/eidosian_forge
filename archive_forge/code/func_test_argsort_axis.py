import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_argsort_axis(self):
    ser = Series(range(3))
    msg = 'No axis named 2 for object type Series'
    with pytest.raises(ValueError, match=msg):
        ser.argsort(axis=2)