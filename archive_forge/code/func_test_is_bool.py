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
def test_is_bool(self):
    assert is_bool(True)
    assert is_bool(False)
    assert is_bool(np.bool_(False))
    assert not is_bool(1)
    assert not is_bool(1.1)
    assert not is_bool(1 + 3j)
    assert not is_bool(np.int64(1))
    assert not is_bool(np.float64(1.1))
    assert not is_bool(np.complex128(1 + 3j))
    assert not is_bool(np.nan)
    assert not is_bool(None)
    assert not is_bool('x')
    assert not is_bool(datetime(2011, 1, 1))
    assert not is_bool(np.datetime64('2011-01-01'))
    assert not is_bool(Timestamp('2011-01-01'))
    assert not is_bool(Timestamp('2011-01-01', tz='US/Eastern'))
    assert not is_bool(timedelta(1000))
    assert not is_bool(np.timedelta64(1, 'D'))
    assert not is_bool(Timedelta('1 days'))