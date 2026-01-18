from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_memory_usage_empty_no_warning():
    df = DataFrame(index=['a', 'b'])
    with tm.assert_produces_warning(None):
        result = df.memory_usage()
    expected = Series(16 if IS64 else 8, index=['Index'])
    tm.assert_series_equal(result, expected)