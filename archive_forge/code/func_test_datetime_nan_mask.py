import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_datetime_nan_mask():
    result = cut(date_range('20130102', periods=5), bins=date_range('20130101', periods=2))
    mask = result.categories.isna()
    tm.assert_numpy_array_equal(mask, np.array([False]))
    mask = result.isna()
    tm.assert_numpy_array_equal(mask, np.array([False, True, True, True, True]))