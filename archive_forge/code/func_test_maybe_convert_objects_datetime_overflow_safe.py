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
@pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
def test_maybe_convert_objects_datetime_overflow_safe(self, dtype):
    stamp = datetime(2363, 10, 4)
    if dtype == 'timedelta64[ns]':
        stamp = stamp - datetime(1970, 1, 1)
    arr = np.array([stamp], dtype=object)
    out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(out, arr)