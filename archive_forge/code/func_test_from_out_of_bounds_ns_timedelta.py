import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('cls', [timedelta, np.timedelta64])
def test_from_out_of_bounds_ns_timedelta(self, constructor, cls, request, box, frame_or_series):
    if box is list or (frame_or_series is Series and box is dict):
        mark = pytest.mark.xfail(reason='TimedeltaArray constructor has been updated to cast td64 to non-nano, but TimedeltaArray._from_sequence has not', strict=True)
        request.applymarker(mark)
    scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
    exp_dtype = 'm8[us]'
    if cls is np.timedelta64:
        scalar = np.timedelta64(scalar, 'D')
        exp_dtype = 'm8[s]'
    result = constructor(scalar)
    item = get1(result)
    dtype = tm.get_dtype(result)
    assert type(item) is Timedelta
    assert item.asm8.dtype == exp_dtype
    assert dtype == exp_dtype