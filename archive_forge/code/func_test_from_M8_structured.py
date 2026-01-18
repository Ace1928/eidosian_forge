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
def test_from_M8_structured(self):
    dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
    arr = np.array(dates, dtype=[('Date', 'M8[us]'), ('Forecasting', 'M8[us]')])
    df = DataFrame(arr)
    assert df['Date'][0] == dates[0][0]
    assert df['Forecasting'][0] == dates[0][1]
    s = Series(arr['Date'])
    assert isinstance(s[0], Timestamp)
    assert s[0] == dates[0][0]