import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_hash_vs_equality(self, dtype):
    dtype2 = IntervalDtype('int64', 'right')
    dtype3 = IntervalDtype(dtype2)
    assert dtype == dtype2
    assert dtype2 == dtype
    assert dtype3 == dtype
    assert dtype is not dtype2
    assert dtype2 is not dtype3
    assert dtype3 is not dtype
    assert hash(dtype) == hash(dtype2)
    assert hash(dtype) == hash(dtype3)
    dtype1 = IntervalDtype('interval')
    dtype2 = IntervalDtype(dtype1)
    dtype3 = IntervalDtype('interval')
    assert dtype2 == dtype1
    assert dtype2 == dtype2
    assert dtype2 == dtype3
    assert dtype2 is not dtype1
    assert dtype2 is dtype2
    assert dtype2 is not dtype3
    assert hash(dtype2) == hash(dtype1)
    assert hash(dtype2) == hash(dtype2)
    assert hash(dtype2) == hash(dtype3)