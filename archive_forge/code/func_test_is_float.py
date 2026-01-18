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
def test_is_float(self):
    assert is_float(1.1)
    assert is_float(np.float64(1.1))
    assert is_float(np.nan)
    assert not is_float(True)
    assert not is_float(1)
    assert not is_float(1 + 3j)
    assert not is_float(False)
    assert not is_float(np.bool_(False))
    assert not is_float(np.int64(1))
    assert not is_float(np.complex128(1 + 3j))
    assert not is_float(None)
    assert not is_float('x')
    assert not is_float(datetime(2011, 1, 1))
    assert not is_float(np.datetime64('2011-01-01'))
    assert not is_float(Timestamp('2011-01-01'))
    assert not is_float(Timestamp('2011-01-01', tz='US/Eastern'))
    assert not is_float(timedelta(1000))
    assert not is_float(np.timedelta64(1, 'D'))
    assert not is_float(Timedelta('1 days'))