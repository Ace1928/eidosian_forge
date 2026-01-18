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
def test_arith_mixed(self):
    left = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3]})
    result = left + left
    expected = DataFrame({'A': ['aa', 'bb', 'cc'], 'B': [2, 4, 6]})
    tm.assert_frame_equal(result, expected)