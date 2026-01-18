from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_with_name_as_column_name():
    data = {'name': ['foo', 'bar']}
    df = DataFrame(data)
    result = df.agg({'name': 'count'})
    expected = Series({'name': 2})
    tm.assert_series_equal(result, expected)
    result = df['name'].agg({'name': 'count'})
    expected = Series({'name': 2}, name='name')
    tm.assert_series_equal(result, expected)