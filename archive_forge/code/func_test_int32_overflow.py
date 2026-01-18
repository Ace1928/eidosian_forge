from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_int32_overflow():
    B = np.concatenate((np.arange(10000), np.arange(10000), np.arange(5000)))
    A = np.arange(25000)
    df = DataFrame({'A': A, 'B': B, 'C': A, 'D': B, 'E': np.random.default_rng(2).standard_normal(25000)})
    left = df.groupby(['A', 'B', 'C', 'D']).sum()
    right = df.groupby(['D', 'C', 'B', 'A']).sum()
    assert len(left) == len(right)