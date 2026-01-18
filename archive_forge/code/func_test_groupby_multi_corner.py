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
def test_groupby_multi_corner(df):
    df = df.copy()
    df['bad'] = np.nan
    agged = df.groupby(['A', 'B']).mean()
    expected = df.groupby(['A', 'B']).mean()
    expected['bad'] = np.nan
    tm.assert_frame_equal(agged, expected)