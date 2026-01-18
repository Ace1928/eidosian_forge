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
def test_groupby_nonstring_columns():
    df = DataFrame([np.arange(10) for x in range(10)])
    grouped = df.groupby(0)
    result = grouped.mean()
    expected = df.groupby(df[0]).mean()
    tm.assert_frame_equal(result, expected)