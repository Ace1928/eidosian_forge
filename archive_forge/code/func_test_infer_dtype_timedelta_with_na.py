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
@pytest.mark.parametrize('delta', [Timedelta('1 days'), np.timedelta64(1, 'D'), timedelta(1)])
def test_infer_dtype_timedelta_with_na(self, na_value, delta):
    arr = np.array([na_value, delta])
    assert lib.infer_dtype(arr, skipna=True) == 'timedelta'
    arr = np.array([na_value, delta, na_value])
    assert lib.infer_dtype(arr, skipna=True) == 'timedelta'