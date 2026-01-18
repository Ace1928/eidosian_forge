from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_categorical_invalid_conversions(self):
    cat = Categorical([f'{i} - {i + 499}' for i in range(0, 10000, 500)])
    ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
    ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)
    msg = "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' not understood"
    with pytest.raises(TypeError, match=msg):
        ser.astype(Categorical)
    with pytest.raises(TypeError, match=msg):
        ser.astype('object').astype(Categorical)