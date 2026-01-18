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
@pytest.mark.parametrize('target', [1, 'cat', [1, 2], np.array([]), (1, 3), {1: 2}])
def test_inplace_no_assignment(self, target):
    expression = '1 + 2'
    assert self.eval(expression, target=target, inplace=False) == 3
    msg = 'Cannot operate inplace if there is no assignment'
    with pytest.raises(ValueError, match=msg):
        self.eval(expression, target=target, inplace=True)