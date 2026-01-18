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
def test_constructor_with_datetimes4(self):
    dr = date_range('20130101', periods=3)
    df = DataFrame({'value': dr})
    assert df.iat[0, 0].tz is None
    dr = date_range('20130101', periods=3, tz='UTC')
    df = DataFrame({'value': dr})
    assert str(df.iat[0, 0].tz) == 'UTC'
    dr = date_range('20130101', periods=3, tz='US/Eastern')
    df = DataFrame({'value': dr})
    assert str(df.iat[0, 0].tz) == 'US/Eastern'