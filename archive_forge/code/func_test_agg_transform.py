from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_transform(axis, float_frame):
    other_axis = 1 if axis in {0, 'index'} else 0
    with np.errstate(all='ignore'):
        f_abs = np.abs(float_frame)
        f_sqrt = np.sqrt(float_frame)
        expected = f_sqrt.copy()
        result = float_frame.apply(np.sqrt, axis=axis)
        tm.assert_frame_equal(result, expected)
        result = float_frame.apply([np.sqrt], axis=axis)
        expected = f_sqrt.copy()
        if axis in {0, 'index'}:
            expected.columns = MultiIndex.from_product([float_frame.columns, ['sqrt']])
        else:
            expected.index = MultiIndex.from_product([float_frame.index, ['sqrt']])
        tm.assert_frame_equal(result, expected)
        result = float_frame.apply([np.abs, np.sqrt], axis=axis)
        expected = zip_frames([f_abs, f_sqrt], axis=other_axis)
        if axis in {0, 'index'}:
            expected.columns = MultiIndex.from_product([float_frame.columns, ['absolute', 'sqrt']])
        else:
            expected.index = MultiIndex.from_product([float_frame.index, ['absolute', 'sqrt']])
        tm.assert_frame_equal(result, expected)