import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_dict_with_tzaware_scalar(self):
    dt = Timestamp('2019-11-03 01:00:00-0700').tz_convert('America/Los_Angeles')
    dt = dt.as_unit('ns')
    df = DataFrame({'dt': dt}, index=[0])
    expected = DataFrame({'dt': [dt]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'dt': dt, 'value': [1]})
    expected = DataFrame({'dt': [dt], 'value': [1]})
    tm.assert_frame_equal(df, expected)