from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('val_in, index, val_out', [([1.0, 2.0, 3.0, 4.0, 5.0], ['foo', 'foo', 'bar', 'baz', 'blah'], [3.0, 4.0, 5.0, 3.0]), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['foo', 'foo', 'bar', 'baz', 'blah', 'blah'], [3.0, 4.0, 11.0, 3.0])])
def test_groupby_index_name_in_index_content(val_in, index, val_out):
    series = Series(data=val_in, name='values', index=Index(index, name='blah'))
    result = series.groupby('blah').sum()
    expected = Series(data=val_out, name='values', index=Index(['bar', 'baz', 'blah', 'foo'], name='blah'))
    tm.assert_series_equal(result, expected)
    result = series.to_frame().groupby('blah').sum()
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)