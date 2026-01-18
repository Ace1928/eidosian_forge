import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_eval_object_dtype_binop(self):
    df = DataFrame({'a1': ['Y', 'N']})
    res = df.eval("c = ((a1 == 'Y') & True)")
    expected = DataFrame({'a1': ['Y', 'N'], 'c': [True, False]})
    tm.assert_frame_equal(res, expected)