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
def test_is_integer(self):
    assert is_integer(1)
    assert is_integer(np.int64(1))
    assert not is_integer(True)
    assert not is_integer(1.1)
    assert not is_integer(1 + 3j)
    assert not is_integer(False)
    assert not is_integer(np.bool_(False))
    assert not is_integer(np.float64(1.1))
    assert not is_integer(np.complex128(1 + 3j))
    assert not is_integer(np.nan)
    assert not is_integer(None)
    assert not is_integer('x')
    assert not is_integer(datetime(2011, 1, 1))
    assert not is_integer(np.datetime64('2011-01-01'))
    assert not is_integer(Timestamp('2011-01-01'))
    assert not is_integer(Timestamp('2011-01-01', tz='US/Eastern'))
    assert not is_integer(timedelta(1000))
    assert not is_integer(Timedelta('1 days'))
    assert not is_integer(np.timedelta64(1, 'D'))