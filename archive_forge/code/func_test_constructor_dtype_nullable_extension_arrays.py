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
@pytest.mark.parametrize('data,input_dtype,expected_dtype', (([True, False, None], 'boolean', pd.BooleanDtype), ([1.0, 2.0, None], 'Float64', pd.Float64Dtype), ([1, 2, None], 'Int64', pd.Int64Dtype), (['a', 'b', 'c'], 'string', pd.StringDtype)))
def test_constructor_dtype_nullable_extension_arrays(self, data, input_dtype, expected_dtype):
    df = DataFrame({'a': data}, dtype=input_dtype)
    assert df['a'].dtype == expected_dtype()