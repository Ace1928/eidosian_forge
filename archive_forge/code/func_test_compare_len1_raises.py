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
def test_compare_len1_raises(self, arr1d):
    arr = arr1d
    idx = self.index_cls(arr)
    with pytest.raises(ValueError, match='Lengths must match'):
        arr == arr[:1]
    with pytest.raises(ValueError, match='Lengths must match'):
        idx <= idx[[0]]