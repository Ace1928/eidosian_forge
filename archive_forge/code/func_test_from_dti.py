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
def test_from_dti(self, arr1d):
    arr = arr1d
    dti = self.index_cls(arr1d)
    assert list(dti) == list(arr)
    dti2 = pd.Index(arr)
    assert isinstance(dti2, DatetimeIndex)
    assert list(dti2) == list(arr)