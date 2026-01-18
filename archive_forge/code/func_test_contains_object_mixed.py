from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_contains_object_mixed():
    mixed = Series(np.array(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0], dtype=object))
    result = mixed.str.contains('o')
    expected = Series(np.array([False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan], dtype=np.object_))
    tm.assert_series_equal(result, expected)