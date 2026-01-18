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
def test_groupby_mixed_type_columns():
    df = DataFrame([[0, 1, 2]], columns=['A', 'B', 0])
    expected = DataFrame([[1, 2]], columns=['B', 0], index=Index([0], name='A'))
    result = df.groupby('A').first()
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').sum()
    tm.assert_frame_equal(result, expected)