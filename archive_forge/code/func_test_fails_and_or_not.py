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
@pytest.mark.parametrize('expr', ['df > 2 and df > 3', 'df > 2 or df > 3', 'not df > 2'])
def test_fails_and_or_not(self, expr, engine, parser):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    if parser == 'python':
        msg = "'BoolOp' nodes are not implemented"
        if 'not' in expr:
            msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(expr, local_dict={'df': df}, parser=parser, engine=engine)
    else:
        pd.eval(expr, local_dict={'df': df}, parser=parser, engine=engine)