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
@pytest.mark.parametrize('op', expr.CMP_OPS_SYMS)
def test_compound_invert_op(self, op, lhs, rhs, request, engine, parser):
    if parser == 'python' and op in ['in', 'not in']:
        msg = "'(In|NotIn)' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            ex = f'~(lhs {op} rhs)'
            pd.eval(ex, engine=engine, parser=parser)
        return
    if is_float(lhs) and (not is_float(rhs)) and (op in ['in', 'not in']) and (engine == 'python') and (parser == 'pandas'):
        mark = pytest.mark.xfail(reason='Looks like expected is negative, unclear whether expected is incorrect or result is incorrect')
        request.applymarker(mark)
    skip_these = ['in', 'not in']
    ex = f'~(lhs {op} rhs)'
    msg = '|'.join(["only list-like( or dict-like)? objects are allowed to be passed to (DataFrame\\.)?isin\\(\\), you passed a (`|')float(`|')", "argument of type 'float' is not iterable"])
    if is_scalar(rhs) and op in skip_these:
        with pytest.raises(TypeError, match=msg):
            pd.eval(ex, engine=engine, parser=parser, local_dict={'lhs': lhs, 'rhs': rhs})
    else:
        if is_scalar(lhs) and is_scalar(rhs):
            lhs, rhs = (np.array([x]) for x in (lhs, rhs))
        expected = _eval_single_bin(lhs, op, rhs, engine)
        if is_scalar(expected):
            expected = not expected
        else:
            expected = ~expected
        result = pd.eval(ex, engine=engine, parser=parser)
        tm.assert_almost_equal(expected, result)