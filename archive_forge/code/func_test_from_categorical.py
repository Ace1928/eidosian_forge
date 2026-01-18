from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_from_categorical(self):
    tdi = timedelta_range(1, periods=5)
    cat = pd.Categorical(tdi)
    result = TimedeltaIndex(cat)
    tm.assert_index_equal(result, tdi)
    ci = pd.CategoricalIndex(tdi)
    result = TimedeltaIndex(ci)
    tm.assert_index_equal(result, tdi)