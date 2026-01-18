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
@td.skip_if_no('numexpr')
def test_invalid_parser():
    msg = "Invalid parser 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval('x + y', local_dict={'x': 1, 'y': 2}, parser='asdf')