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
@pytest.mark.parametrize('df, col_dtype', [(DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list('ab')), 'float64'), (DataFrame([[1.0, 'b'], [4.0, 'b']], columns=list('ab')).astype({'b': object}), 'object')])
def test_dataframe_operation_with_non_numeric_types(df, col_dtype):
    expected = DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list('ab'))
    expected = expected.astype({'b': col_dtype})
    result = df + Series([-1.0], index=list('a'))
    tm.assert_frame_equal(result, expected)