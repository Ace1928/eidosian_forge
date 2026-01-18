from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_int64_nocopy(self):
    arr = np.arange(10, dtype=np.int64)
    tdi = TimedeltaIndex(arr, copy=False)
    assert tdi._data._ndarray.base is arr