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
def test_constructor_dict_datetime64_index(self):
    dates_as_str = ['1984-02-19', '1988-11-06', '1989-12-03', '1990-03-15']

    def create_data(constructor):
        return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}
    data_datetime64 = create_data(np.datetime64)
    data_datetime = create_data(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data_Timestamp = create_data(Timestamp)
    expected = DataFrame([{0: 0, 1: None, 2: None, 3: None}, {0: None, 1: 2, 2: None, 3: None}, {0: None, 1: None, 2: 4, 3: None}, {0: None, 1: None, 2: None, 3: 6}], index=[Timestamp(dt) for dt in dates_as_str])
    result_datetime64 = DataFrame(data_datetime64)
    result_datetime = DataFrame(data_datetime)
    result_Timestamp = DataFrame(data_Timestamp)
    tm.assert_frame_equal(result_datetime64, expected)
    tm.assert_frame_equal(result_datetime, expected)
    tm.assert_frame_equal(result_Timestamp, expected)