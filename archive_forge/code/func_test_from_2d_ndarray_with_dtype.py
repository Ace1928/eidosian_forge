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
def test_from_2d_ndarray_with_dtype(self):
    array_dim2 = np.arange(10).reshape((5, 2))
    df = DataFrame(array_dim2, dtype='datetime64[ns, UTC]')
    expected = DataFrame(array_dim2).astype('datetime64[ns, UTC]')
    tm.assert_frame_equal(df, expected)