from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_getitem_float_slice_floatindex(self, float_numpy_dtype):
    dtype = float_numpy_dtype
    ser = Series(np.random.default_rng(2).random(10), index=np.arange(10, 20, dtype=dtype))
    assert len(ser.loc[12.0:]) == 8
    assert len(ser.loc[12.5:]) == 7
    idx = np.arange(10, 20, dtype=dtype)
    idx[2] = 12.2
    ser.index = idx
    assert len(ser.loc[12.0:]) == 8
    assert len(ser.loc[12.5:]) == 7