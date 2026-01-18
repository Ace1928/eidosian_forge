from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('op', ['apply', 'agg'])
def test_apply_nested_result_axis_1(op):

    def apply_list(row):
        return [2 * row['A'], 2 * row['C'], 2 * row['B']]
    df = DataFrame(np.zeros((4, 4)), columns=list('ABCD'))
    result = getattr(df, op)(apply_list, axis=1)
    expected = Series([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tm.assert_series_equal(result, expected)