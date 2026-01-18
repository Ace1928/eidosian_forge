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
@pytest.mark.parametrize('express', ['a + @b', '@a + b', '@a + @b'])
def test_invalid_local_variable_reference(engine, parser, express):
    a, b = (1, 2)
    if parser != 'pandas':
        with pytest.raises(SyntaxError, match="The '@' prefix is only"):
            pd.eval(express, engine=engine, parser=parser)
    else:
        with pytest.raises(SyntaxError, match="The '@' prefix is not"):
            pd.eval(express, engine=engine, parser=parser)