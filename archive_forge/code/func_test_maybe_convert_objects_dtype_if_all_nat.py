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
def test_maybe_convert_objects_dtype_if_all_nat(self):
    arr = np.array([pd.NaT, pd.NaT], dtype=object)
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, arr)
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('timedelta64[ns]'))
    exp = np.array(['NaT', 'NaT'], dtype='timedelta64[ns]')
    tm.assert_numpy_array_equal(out, exp)
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True, dtype_if_all_nat=np.dtype('datetime64[ns]'))
    exp = np.array(['NaT', 'NaT'], dtype='datetime64[ns]')
    tm.assert_numpy_array_equal(out, exp)