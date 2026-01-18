import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_multi_nan_indexing(self):
    df = DataFrame({'a': ['R1', 'R2', np.nan, 'R4'], 'b': ['C1', 'C2', 'C3', 'C4'], 'c': [10, 15, np.nan, 20]})
    result = df.set_index(['a', 'b'], drop=False)
    expected = DataFrame({'a': ['R1', 'R2', np.nan, 'R4'], 'b': ['C1', 'C2', 'C3', 'C4'], 'c': [10, 15, np.nan, 20]}, index=[Index(['R1', 'R2', np.nan, 'R4'], name='a'), Index(['C1', 'C2', 'C3', 'C4'], name='b')])
    tm.assert_frame_equal(result, expected)