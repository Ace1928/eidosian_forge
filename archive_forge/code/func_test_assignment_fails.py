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
def test_assignment_fails(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=list('abc'))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    expr1 = 'df = df2'
    msg = 'cannot assign without a target object'
    with pytest.raises(ValueError, match=msg):
        self.eval(expr1, local_dict={'df': df, 'df2': df2})