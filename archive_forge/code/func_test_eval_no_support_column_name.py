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
@pytest.mark.parametrize('column', DEFAULT_GLOBALS.keys())
def test_eval_no_support_column_name(request, column):
    if column in ['True', 'False', 'inf', 'Inf']:
        request.applymarker(pytest.mark.xfail(raises=KeyError, reason=f'GH 47859 DataFrame eval not supported with {column}'))
    df = DataFrame(np.random.default_rng(2).integers(0, 100, size=(10, 2)), columns=[column, 'col1'])
    expected = df[df[column] > 6]
    result = df.query(f'{column}>6')
    tm.assert_frame_equal(result, expected)