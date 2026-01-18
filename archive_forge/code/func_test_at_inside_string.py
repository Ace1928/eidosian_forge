import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_at_inside_string(self, engine, parser):
    skip_if_no_pandas_parser(parser)
    c = 1
    df = DataFrame({'a': ['a', 'a', 'b', 'b', '@c', '@c']})
    result = df.query('a == "@c"', engine=engine, parser=parser)
    expected = df[df.a == '@c']
    tm.assert_frame_equal(result, expected)