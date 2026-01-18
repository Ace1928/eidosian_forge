from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_infer_output_shape_columns():
    df = DataFrame({'number': [1.0, 2.0], 'string': ['foo', 'bar'], 'datetime': [Timestamp('2017-11-29 03:30:00'), Timestamp('2017-11-29 03:45:00')]})
    result = df.apply(lambda row: (row.number, row.string), axis=1)
    expected = Series([(t.number, t.string) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)