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
def test_constructor_list_of_lists(self, using_infer_string):
    df = DataFrame(data=[[1, 'a'], [2, 'b']], columns=['num', 'str'])
    assert is_integer_dtype(df['num'])
    assert df['str'].dtype == np.object_ if not using_infer_string else 'string'
    expected = DataFrame({0: np.arange(10)})
    data = [np.array(x) for x in range(10)]
    result = DataFrame(data)
    tm.assert_frame_equal(result, expected)