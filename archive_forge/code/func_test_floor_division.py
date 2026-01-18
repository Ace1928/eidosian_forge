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
def test_floor_division(self, lhs, rhs, engine, parser):
    ex = 'lhs // rhs'
    if engine == 'python':
        res = pd.eval(ex, engine=engine, parser=parser)
        expected = lhs // rhs
        tm.assert_equal(res, expected)
    else:
        msg = "unsupported operand type\\(s\\) for //: 'VariableNode' and 'VariableNode'"
        with pytest.raises(TypeError, match=msg):
            pd.eval(ex, local_dict={'lhs': lhs, 'rhs': rhs}, engine=engine, parser=parser)