import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_multiindex_from_tuples_with_nan(self):
    result = MultiIndex.from_tuples([('a', 'b', 'c'), np.nan, ('d', '', '')])
    expected = MultiIndex.from_tuples([('a', 'b', 'c'), (np.nan, np.nan, np.nan), ('d', '', '')])
    tm.assert_index_equal(result, expected)