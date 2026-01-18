import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_bins_datetime_intervalindex():
    bins = interval_range(Timestamp('2022-02-25'), Timestamp('2022-02-27'), freq='1D')
    result = cut(Series([Timestamp('2022-02-26')]).astype('M8[ns]'), bins=bins)
    expected = Categorical.from_codes([0], bins, ordered=True)
    tm.assert_categorical_equal(result.array, expected)