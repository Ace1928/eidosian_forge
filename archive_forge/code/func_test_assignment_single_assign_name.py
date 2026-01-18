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
def test_assignment_single_assign_name(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=list('ab'))
    a = 1
    old_a = df.a.copy()
    df.eval('a = a + b', inplace=True)
    result = old_a + df.b
    tm.assert_series_equal(result, df.a, check_names=False)
    assert result.name is None