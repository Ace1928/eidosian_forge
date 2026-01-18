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
@pytest.mark.parametrize('dict_type', [dict, OrderedDict])
def test_constructor_ordered_dict_conflicting_orders(self, dict_type):
    row_one = dict_type()
    row_one['b'] = 2
    row_one['a'] = 1
    row_two = dict_type()
    row_two['a'] = 1
    row_two['b'] = 2
    row_three = {'b': 2, 'a': 1}
    expected = DataFrame([[2, 1], [2, 1]], columns=['b', 'a'])
    result = DataFrame([row_one, row_two])
    tm.assert_frame_equal(result, expected)
    expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=['b', 'a'])
    result = DataFrame([row_one, row_two, row_three])
    tm.assert_frame_equal(result, expected)