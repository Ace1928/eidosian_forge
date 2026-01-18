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
def test_keyword_arg(self, engine, parser):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
    msg = 'Function "sin" does not support keyword arguments'
    with pytest.raises(TypeError, match=msg):
        df.eval('sin(x=a)', engine=engine, parser=parser)