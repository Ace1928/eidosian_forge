from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
@pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
def test_df_flex_cmp_constant_return_types_empty(self, opname):
    df = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
    const = 2
    empty = df.iloc[:0]
    result = getattr(empty, opname)(const).dtypes.value_counts()
    tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))