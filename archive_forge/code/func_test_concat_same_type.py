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
def test_concat_same_type(self, arr1d):
    arr = arr1d
    idx = self.index_cls(arr)
    idx = idx.insert(0, NaT)
    arr = arr1d
    result = arr._concat_same_type([arr[:-1], arr[1:], arr])
    arr2 = arr.astype(object)
    expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))
    tm.assert_index_equal(self.index_cls(result), expected)