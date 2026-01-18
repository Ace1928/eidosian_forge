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
def test_astype_bool_missing_to_categorical(self):
    ser = Series([True, False, np.nan])
    assert ser.dtypes == np.object_
    result = ser.astype(CategoricalDtype(categories=[True, False]))
    expected = Series(Categorical([True, False, np.nan], categories=[True, False]))
    tm.assert_series_equal(result, expected)