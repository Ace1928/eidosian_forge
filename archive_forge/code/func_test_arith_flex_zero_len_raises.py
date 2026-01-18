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
def test_arith_flex_zero_len_raises(self):
    ser_len0 = Series([], dtype=object)
    df_len0 = DataFrame(columns=['A', 'B'])
    df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    with pytest.raises(NotImplementedError, match='fill_value'):
        df.add(ser_len0, fill_value='E')
    with pytest.raises(NotImplementedError, match='fill_value'):
        df_len0.sub(df['A'], axis=None, fill_value=3)