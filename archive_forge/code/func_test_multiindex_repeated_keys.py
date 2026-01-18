import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_multiindex_repeated_keys(self):
    tm.assert_series_equal(Series([1, 2], MultiIndex.from_arrays([['a', 'b']])).loc[['a', 'a', 'b', 'b']], Series([1, 1, 2, 2], MultiIndex.from_arrays([['a', 'a', 'b', 'b']])))