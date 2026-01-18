from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_adding_new_conditional_column() -> None:
    df = DataFrame({'x': [1]})
    df.loc[df['x'] == 1, 'y'] = '1'
    expected = DataFrame({'x': [1], 'y': ['1']})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'x': [1]})
    value = lambda x: x
    df.loc[df['x'] == 1, 'y'] = value
    expected = DataFrame({'x': [1], 'y': [value]})
    tm.assert_frame_equal(df, expected)