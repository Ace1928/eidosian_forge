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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('index_name', ['index', 'columns'])
@pytest.mark.parametrize('c_idx_type', index_types)
@pytest.mark.parametrize('r_idx_type', lhs_index_types)
def test_basic_frame_series_alignment(self, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)), index=idx_func_dict[r_idx_type](10), columns=idx_func_dict[c_idx_type](10))
    index = getattr(df, index_name)
    s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
    if should_warn(df.index, s.index):
        with tm.assert_produces_warning(RuntimeWarning):
            res = pd.eval('df + s', engine=engine, parser=parser)
    else:
        res = pd.eval('df + s', engine=engine, parser=parser)
    if r_idx_type == 'dt' or c_idx_type == 'dt':
        expected = df.add(s) if engine == 'numexpr' else df + s
    else:
        expected = df + s
    tm.assert_frame_equal(res, expected)