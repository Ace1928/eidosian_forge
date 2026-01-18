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
def test_construct_ndarray_with_nas_and_int_dtype(self):
    arr = np.array([[1, np.nan], [2, 3]])
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    with pytest.raises(IntCastingNaNError, match=msg):
        DataFrame(arr, dtype='i8')
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr[0], dtype='i8', name=0)