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
def test_groupby_aggregation_non_numeric_dtype():
    df = DataFrame([['M', [1]], ['M', [1]], ['W', [10]], ['W', [20]]], columns=['MW', 'v'])
    expected = DataFrame({'v': [[1, 1], [10, 20]]}, index=Index(['M', 'W'], dtype='object', name='MW'))
    gb = df.groupby(by=['MW'])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)