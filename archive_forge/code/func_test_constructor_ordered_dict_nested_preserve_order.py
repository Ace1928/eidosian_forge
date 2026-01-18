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
def test_constructor_ordered_dict_nested_preserve_order(self):
    nested1 = OrderedDict([('b', 1), ('a', 2)])
    nested2 = OrderedDict([('b', 2), ('a', 5)])
    data = OrderedDict([('col2', nested1), ('col1', nested2)])
    result = DataFrame(data)
    data = {'col2': [1, 2], 'col1': [2, 5]}
    expected = DataFrame(data=data, index=['b', 'a'])
    tm.assert_frame_equal(result, expected)