from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string):
    s1 = Series(['a', 'b', 'c'])
    s2 = Series(['a', 'b'])
    s3 = Series(['a', 'b', 'c', 'd'])
    s4 = Series([], dtype=object if not using_infer_string else 'string[pyarrow_numpy]')
    result = concat([s1, s2, s3, s4], sort=False, join='outer', ignore_index=False, axis=1)
    expected = DataFrame([['a'] * 3 + [np.nan], ['b'] * 3 + [np.nan], ['c', np.nan] * 2, [np.nan] * 2 + ['d'] + [np.nan]], dtype=object if not using_infer_string else 'string[pyarrow_numpy]')
    tm.assert_frame_equal(result, expected, check_index_type=True, check_column_type=True)