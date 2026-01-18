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
def test_constructor_with_datetimes3(self):
    tz = pytz.timezone('US/Eastern')
    dt = tz.localize(datetime(2012, 1, 1))
    df = DataFrame({'End Date': dt}, index=[0])
    assert df.iat[0, 0] == dt
    tm.assert_series_equal(df.dtypes, Series({'End Date': 'datetime64[us, US/Eastern]'}, dtype=object))
    df = DataFrame([{'End Date': dt}])
    assert df.iat[0, 0] == dt
    tm.assert_series_equal(df.dtypes, Series({'End Date': 'datetime64[ns, US/Eastern]'}, dtype=object))