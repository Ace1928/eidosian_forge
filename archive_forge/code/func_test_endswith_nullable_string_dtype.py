from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('na', [None, True, False])
def test_endswith_nullable_string_dtype(nullable_string_dtype, na):
    values = Series(['om', None, 'foo_nom', 'nom', 'bar_foo', None, 'foo', 'regex', 'rege.'], dtype=nullable_string_dtype)
    result = values.str.endswith('foo', na=na)
    exp = Series([False, na, False, False, True, na, True, False, False], dtype='boolean')
    tm.assert_series_equal(result, exp)
    result = values.str.endswith('rege.', na=na)
    exp = Series([False, na, False, False, False, na, False, False, True], dtype='boolean')
    tm.assert_series_equal(result, exp)