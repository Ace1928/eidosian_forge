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
def test_nonagg_agg():
    df = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 2, 1]})
    g = df.groupby('a')
    result = g.agg(['cumsum'])
    result.columns = result.columns.droplevel(-1)
    expected = g.agg('cumsum')
    tm.assert_frame_equal(result, expected)