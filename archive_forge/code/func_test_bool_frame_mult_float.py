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
def test_bool_frame_mult_float():
    df = DataFrame(True, list('ab'), list('cd'))
    result = df * 1.0
    expected = DataFrame(np.ones((2, 2)), list('ab'), list('cd'))
    tm.assert_frame_equal(result, expected)