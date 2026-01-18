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
def test_df_bool_mul_int(self):
    df = DataFrame([[False, True], [False, False]])
    result = df * 1
    kinds = result.dtypes.apply(lambda x: x.kind)
    assert (kinds == 'i').all()
    result = 1 * df
    kinds = result.dtypes.apply(lambda x: x.kind)
    assert (kinds == 'i').all()