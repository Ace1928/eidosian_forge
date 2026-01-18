from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture
def tda(self, unit):
    arr = np.arange(5, dtype=np.int64).view(f'm8[{unit}]')
    return TimedeltaArray._simple_new(arr, dtype=arr.dtype)