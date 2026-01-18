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
def test_floating_values_integer_dtype(self):
    arr = np.random.default_rng(2).standard_normal((10, 5))
    msg = 'Trying to coerce float values to integers'
    with pytest.raises(ValueError, match=msg):
        DataFrame(arr, dtype='i8')
    df = DataFrame(arr.round(), dtype='i8')
    assert (df.dtypes == 'i8').all()
    arr[0, 0] = np.nan
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        DataFrame(arr, dtype='i8')
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr[0], dtype='i8')
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        DataFrame(arr).astype('i8')
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr[0]).astype('i8')