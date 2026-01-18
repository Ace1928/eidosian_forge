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
def test_constructor_ndarray_copy(self, float_frame, using_array_manager, using_copy_on_write):
    if not using_array_manager:
        arr = float_frame.values.copy()
        df = DataFrame(arr)
        arr[5] = 5
        if using_copy_on_write:
            assert not (df.values[5] == 5).all()
        else:
            assert (df.values[5] == 5).all()
        df = DataFrame(arr, copy=True)
        arr[6] = 6
        assert not (df.values[6] == 6).all()
    else:
        arr = float_frame.values.copy()
        df = DataFrame(arr)
        assert df._mgr.arrays[0].flags.c_contiguous
        arr[0, 0] = 100
        assert df.iloc[0, 0] != 100
        df = DataFrame(arr, copy=False)
        assert not df._mgr.arrays[0].flags.c_contiguous
        arr[0, 0] = 1000
        assert df.iloc[0, 0] == 1000