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
@pytest.mark.parametrize('r1', lhs_index_types)
@pytest.mark.parametrize('c1', index_types)
@pytest.mark.parametrize('r2', index_types)
@pytest.mark.parametrize('c2', index_types)
def test_medium_complex_frame_alignment(self, engine, parser, r1, c1, r2, c2, idx_func_dict):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), index=idx_func_dict[r1](3), columns=idx_func_dict[c1](2))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), index=idx_func_dict[r2](4), columns=idx_func_dict[c2](2))
    df3 = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=idx_func_dict[r2](5), columns=idx_func_dict[c2](2))
    if should_warn(df.index, df2.index, df3.index):
        with tm.assert_produces_warning(RuntimeWarning):
            res = pd.eval('df + df2 + df3', engine=engine, parser=parser)
    else:
        res = pd.eval('df + df2 + df3', engine=engine, parser=parser)
    tm.assert_frame_equal(res, df + df2 + df3)