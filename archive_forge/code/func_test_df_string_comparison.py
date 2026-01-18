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
def test_df_string_comparison(self):
    df = DataFrame([{'a': 1, 'b': 'foo'}, {'a': 2, 'b': 'bar'}])
    mask_a = df.a > 1
    tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
    tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])
    mask_b = df.b == 'foo'
    tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
    tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])