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
@pytest.mark.parametrize('invalid_target', [1, 'cat', [1, 2], np.array([]), (1, 3)])
def test_cannot_item_assign(self, invalid_target):
    msg = 'Cannot assign expression output to target'
    expression = 'a = 1 + 2'
    with pytest.raises(ValueError, match=msg):
        self.eval(expression, target=invalid_target, inplace=True)
    if hasattr(invalid_target, 'copy'):
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=False)