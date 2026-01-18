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
def test_no_warning(self, all_arithmetic_operators):
    df = DataFrame({'A': [0.0, 0.0], 'B': [0.0, None]})
    b = df['B']
    with tm.assert_produces_warning(None):
        getattr(df, all_arithmetic_operators)(b)