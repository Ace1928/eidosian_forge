from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_copy_index_series(self, axis, using_copy_on_write):
    ser = Series([1, 2])
    comb = concat([ser, ser], axis=axis, copy=True)
    if not using_copy_on_write or axis in [0, 'index']:
        assert comb.index is not ser.index
    else:
        assert comb.index is ser.index