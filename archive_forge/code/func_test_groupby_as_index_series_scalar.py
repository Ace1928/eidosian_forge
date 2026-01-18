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
def test_groupby_as_index_series_scalar(df):
    grouped = df.groupby(['A', 'B'], as_index=False)
    result = grouped['C'].agg(len)
    expected = grouped.agg(len).loc[:, ['A', 'B', 'C']]
    tm.assert_frame_equal(result, expected)