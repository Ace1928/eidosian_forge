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
@pytest.mark.parametrize('data', [[[Timestamp('2021-01-01')]], [{'x': Timestamp('2021-01-01')}], {'x': [Timestamp('2021-01-01')]}, {'x': Timestamp('2021-01-01').as_unit('ns')}])
def test_constructor_one_element_data_list(self, data):
    result = DataFrame(data, index=[0, 1, 2], columns=['x'])
    expected = DataFrame({'x': [Timestamp('2021-01-01')] * 3})
    tm.assert_frame_equal(result, expected)