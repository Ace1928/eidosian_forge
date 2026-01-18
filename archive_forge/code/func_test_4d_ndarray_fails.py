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
def test_4d_ndarray_fails(self):
    x = np.random.default_rng(2).standard_normal((3, 4, 5, 6))
    y = Series(np.random.default_rng(2).standard_normal(10))
    msg = 'N-dimensional objects, where N > 2, are not supported with eval'
    with pytest.raises(NotImplementedError, match=msg):
        self.eval('x + y', local_dict={'x': x, 'y': y})