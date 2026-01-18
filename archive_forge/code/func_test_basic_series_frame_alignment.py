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
@pytest.mark.parametrize('index_name', ['index', 'columns'])
@pytest.mark.parametrize('r_idx_type, c_idx_type', list(product(['i', 's'], ['i', 's'])) + [('dt', 'dt')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_basic_series_frame_alignment(self, request, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict):
    if engine == 'numexpr' and parser in ('pandas', 'python') and (index_name == 'index') and (r_idx_type == 'i') and (c_idx_type == 's'):
        reason = f'Flaky column ordering when engine={engine}, parser={parser}, index_name={index_name}, r_idx_type={r_idx_type}, c_idx_type={c_idx_type}'
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 7)), index=idx_func_dict[r_idx_type](10), columns=idx_func_dict[c_idx_type](7))
    index = getattr(df, index_name)
    s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
    if should_warn(s.index, df.index):
        with tm.assert_produces_warning(RuntimeWarning):
            res = pd.eval('s + df', engine=engine, parser=parser)
    else:
        res = pd.eval('s + df', engine=engine, parser=parser)
    if r_idx_type == 'dt' or c_idx_type == 'dt':
        expected = df.add(s) if engine == 'numexpr' else s + df
    else:
        expected = s + df
    tm.assert_frame_equal(res, expected)