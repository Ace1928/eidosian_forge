from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
@pytest.mark.parametrize('op', ['+', '-', '*', '**', '/'])
@pytest.mark.parametrize('dt', [np.float32, np.float64])
@pytest.mark.parametrize('left_right', [('df', '3'), ('3', 'df')])
def test_binop_typecasting(self, engine, parser, op, dt, left_right):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), dtype=dt)
    left, right = left_right
    s = f'{left} {op} {right}'
    res = pd.eval(s, engine=engine, parser=parser)
    assert df.values.dtype == dt
    assert res.values.dtype == dt
    tm.assert_frame_equal(res, eval(s))