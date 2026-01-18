import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_aggregate_empty_key_empty_return():
    df = DataFrame({'a': [1, 1, 2], 'b': [1, 2, 3], 'c': [1, 2, 4]})
    result = df.groupby('a').agg({'b': []})
    expected = DataFrame(columns=MultiIndex(levels=[['b'], []], codes=[[], []]))
    tm.assert_frame_equal(result, expected)