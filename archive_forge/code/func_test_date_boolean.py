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
def test_date_boolean(self, engine, parser):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    df['dates1'] = date_range('1/1/2012', periods=5)
    res = self.eval('df.dates1 < 20130101', local_dict={'df': df}, engine=engine, parser=parser)
    expec = df.dates1 < '20130101'
    tm.assert_series_equal(res, expec, check_names=False)