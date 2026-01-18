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
@pytest.mark.parametrize('na_value', [pd.NaT, np.nan])
@pytest.mark.parametrize('time_stamp', [Timestamp('2011-01-01'), datetime(2011, 1, 1)])
def test_infer_dtype_datetime_with_na(self, na_value, time_stamp):
    arr = np.array([na_value, time_stamp])
    assert lib.infer_dtype(arr, skipna=True) == 'datetime'
    arr = np.array([na_value, time_stamp, na_value])
    assert lib.infer_dtype(arr, skipna=True) == 'datetime'