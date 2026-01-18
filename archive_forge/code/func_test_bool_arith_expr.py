import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_bool_arith_expr(self, frame, parser, engine):
    res = frame.eval('a[a < 1] + b', engine=engine, parser=parser)
    expect = frame.a[frame.a < 1] + frame.b
    tm.assert_series_equal(res, expect)