from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_match_mixed_object():
    mixed = Series(['aBAD_BAD', np.nan, 'BAD_b_BAD', True, datetime.today(), 'foo', None, 1, 2.0])
    result = Series(mixed).str.match('.*(BAD[_]+).*(BAD)')
    expected = Series([True, np.nan, True, np.nan, np.nan, False, None, np.nan, np.nan])
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)