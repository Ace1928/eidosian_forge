import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.parametrize('arith', ('add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'))
@pytest.mark.parametrize('axis', (0, 1))
def test_frame_series_axis(self, axis, arith, _frame, monkeypatch):
    df = _frame
    if axis == 1:
        other = df.iloc[0, :]
    else:
        other = df.iloc[:, 0]
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', 0)
        op_func = getattr(df, arith)
        with option_context('compute.use_numexpr', False):
            expected = op_func(other, axis=axis)
        result = op_func(other, axis=axis)
        tm.assert_frame_equal(expected, result)