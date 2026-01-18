from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_infer_output_shape_listlike_columns():
    df = DataFrame(np.random.default_rng(2).standard_normal((6, 3)), columns=['A', 'B', 'C'])
    result = df.apply(lambda x: [1, 2, 3], axis=1)
    expected = Series([[1, 2, 3] for t in df.itertuples()])
    tm.assert_series_equal(result, expected)
    result = df.apply(lambda x: [1, 2], axis=1)
    expected = Series([[1, 2] for t in df.itertuples()])
    tm.assert_series_equal(result, expected)