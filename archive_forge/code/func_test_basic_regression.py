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
def test_basic_regression():
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])
    data = np.random.default_rng(2).random(1100) * 10.0
    groupings = Series(data)
    grouped = result.groupby(groupings)
    grouped.mean()