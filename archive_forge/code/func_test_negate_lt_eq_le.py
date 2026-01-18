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
def test_negate_lt_eq_le(engine, parser):
    df = DataFrame([[0, 10], [1, 20]], columns=['cat', 'count'])
    expected = df[~(df.cat > 0)]
    result = df.query('~(cat > 0)', engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)
    if parser == 'python':
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query('not (cat > 0)', engine=engine, parser=parser)
    else:
        result = df.query('not (cat > 0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)