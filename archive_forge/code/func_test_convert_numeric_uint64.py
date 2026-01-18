import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_convert_numeric_uint64(self):
    arr = np.array([2 ** 63], dtype=object)
    exp = np.array([2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
    arr = np.array([str(2 ** 63)], dtype=object)
    exp = np.array([2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)
    arr = np.array([np.uint64(2 ** 63)], dtype=object)
    exp = np.array([2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)