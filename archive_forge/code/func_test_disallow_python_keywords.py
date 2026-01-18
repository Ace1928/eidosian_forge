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
def test_disallow_python_keywords(self):
    df = DataFrame([[0, 0, 0]], columns=['foo', 'bar', 'class'])
    msg = 'Python keyword not valid identifier in numexpr query'
    with pytest.raises(SyntaxError, match=msg):
        df.query('class == 0')
    df = DataFrame()
    df.index.name = 'lambda'
    with pytest.raises(SyntaxError, match=msg):
        df.query('lambda == 0')