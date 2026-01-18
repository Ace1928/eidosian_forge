import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dropna_categorical_interval_index(self):
    ii = pd.IntervalIndex.from_breaks([0, 2.78, 3.14, 6.28])
    ci = pd.CategoricalIndex(ii)
    df = DataFrame({'A': list('abc')}, index=ci)
    expected = df
    result = df.dropna()
    tm.assert_frame_equal(result, expected)