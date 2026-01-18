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
def test_constructor_list_of_dict_order(self):
    data = [{'First': 1, 'Second': 4, 'Third': 7, 'Fourth': 10}, {'Second': 5, 'First': 2, 'Fourth': 11, 'Third': 8}, {'Second': 6, 'First': 3, 'Fourth': 12, 'Third': 9, 'YYY': 14, 'XXX': 13}]
    expected = DataFrame({'First': [1, 2, 3], 'Second': [4, 5, 6], 'Third': [7, 8, 9], 'Fourth': [10, 11, 12], 'YYY': [None, None, 14], 'XXX': [None, None, 13]})
    result = DataFrame(data)
    tm.assert_frame_equal(result, expected)