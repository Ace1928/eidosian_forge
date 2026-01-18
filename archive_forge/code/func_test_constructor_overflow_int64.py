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
def test_constructor_overflow_int64(self):
    values = np.array([2 ** 64 - i for i in range(1, 10)], dtype=np.uint64)
    result = DataFrame({'a': values})
    assert result['a'].dtype == np.uint64
    data_scores = [(6311132704823138710, 273), (2685045978526272070, 23), (8921811264899370420, 45), (17019687244989530680, 270), (9930107427299601010, 273)]
    dtype = [('uid', 'u8'), ('score', 'u8')]
    data = np.zeros((len(data_scores),), dtype=dtype)
    data[:] = data_scores
    df_crawls = DataFrame(data)
    assert df_crawls['uid'].dtype == np.uint64