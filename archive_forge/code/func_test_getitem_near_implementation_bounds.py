from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_getitem_near_implementation_bounds(self):
    i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype='i8')
    if self.array_cls is PeriodArray:
        arr = self.array_cls(i8vals, dtype='period[ns]')
    else:
        arr = self.index_cls(i8vals, freq='ns')._data
    arr[0]
    index = pd.Index(arr)
    index[0]
    ser = pd.Series(arr)
    ser[0]