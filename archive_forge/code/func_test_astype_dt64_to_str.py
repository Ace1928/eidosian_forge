from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_dt64_to_str(self):
    dti = date_range('2012-01-01', periods=3)
    result = Series(dti).astype(str)
    expected = Series(['2012-01-01', '2012-01-02', '2012-01-03'], dtype=object)
    tm.assert_series_equal(result, expected)