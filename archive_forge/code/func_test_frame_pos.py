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
@pytest.mark.parametrize('lhs', [DataFrame(np.random.default_rng(2).standard_normal((5, 2))), DataFrame(np.random.default_rng(2).integers(5, size=(5, 2))), DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)])
def test_frame_pos(self, lhs, engine, parser):
    expr = '+lhs'
    expect = lhs
    result = pd.eval(expr, engine=engine, parser=parser)
    tm.assert_frame_equal(expect, result)