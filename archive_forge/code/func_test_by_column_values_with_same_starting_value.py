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
@pytest.mark.parametrize('dtype', [object, pytest.param('string[pyarrow_numpy]', marks=td.skip_if_no('pyarrow'))])
def test_by_column_values_with_same_starting_value(dtype):
    df = DataFrame({'Name': ['Thomas', 'Thomas', 'Thomas John'], 'Credit': [1200, 1300, 900], 'Mood': Series(['sad', 'happy', 'happy'], dtype=dtype)})
    aggregate_details = {'Mood': Series.mode, 'Credit': 'sum'}
    result = df.groupby(['Name']).agg(aggregate_details)
    expected_result = DataFrame({'Mood': [['happy', 'sad'], 'happy'], 'Credit': [2500, 900], 'Name': ['Thomas', 'Thomas John']}).set_index('Name')
    tm.assert_frame_equal(result, expected_result)