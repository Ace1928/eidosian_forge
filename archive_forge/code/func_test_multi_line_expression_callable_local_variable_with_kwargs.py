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
def test_multi_line_expression_callable_local_variable_with_kwargs(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def local_func(a, b):
        return b
    expected = df.copy()
    expected['c'] = expected['a'] * local_func(b=7, a=1)
    expected['d'] = expected['c'] + local_func(b=7, a=1)
    answer = df.eval('\n        c = a * @local_func(b=7, a=1)\n        d = c + @local_func(b=7, a=1)\n        ', inplace=True)
    tm.assert_frame_equal(expected, df)
    assert answer is None