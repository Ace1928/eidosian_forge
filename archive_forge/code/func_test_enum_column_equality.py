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
def test_enum_column_equality():
    Cols = Enum('Cols', 'col1 col2')
    q1 = DataFrame({Cols.col1: [1, 2, 3]})
    q2 = DataFrame({Cols.col1: [1, 2, 3]})
    result = q1[Cols.col1] == q2[Cols.col1]
    expected = Series([True, True, True], name=Cols.col1)
    tm.assert_series_equal(result, expected)