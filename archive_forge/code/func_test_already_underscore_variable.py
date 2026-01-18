import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_already_underscore_variable(self, df):
    res = df.eval('`C_C` + A')
    expect = df['C_C'] + df['A']
    tm.assert_series_equal(res, expect)