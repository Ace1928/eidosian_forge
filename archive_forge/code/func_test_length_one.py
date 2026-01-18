from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
def test_length_one(self):
    """breaks of length one produce an empty IntervalIndex"""
    breaks = [0]
    result = IntervalIndex.from_breaks(breaks)
    expected = IntervalIndex.from_breaks([])
    tm.assert_index_equal(result, expected)