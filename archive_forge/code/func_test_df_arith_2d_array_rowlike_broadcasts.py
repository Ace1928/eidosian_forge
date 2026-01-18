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
def test_df_arith_2d_array_rowlike_broadcasts(self, request, all_arithmetic_operators, using_array_manager):
    opname = all_arithmetic_operators
    if using_array_manager and opname in ('__rmod__', '__rfloordiv__'):
        td.mark_array_manager_not_yet_implemented(request)
    arr = np.arange(6).reshape(3, 2)
    df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
    rowlike = arr[[1], :]
    assert rowlike.shape == (1, df.shape[1])
    exvals = [getattr(df.loc['A'], opname)(rowlike.squeeze()), getattr(df.loc['B'], opname)(rowlike.squeeze()), getattr(df.loc['C'], opname)(rowlike.squeeze())]
    expected = DataFrame(exvals, columns=df.columns, index=df.index)
    result = getattr(df, opname)(rowlike)
    tm.assert_frame_equal(result, expected)