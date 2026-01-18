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
@pytest.mark.parametrize('dtype', tm.STRING_DTYPES + tm.BYTES_DTYPES + tm.OBJECT_DTYPES)
def test_check_dtype_empty_string_column(self, request, dtype, using_array_manager):
    data = DataFrame({'a': [1, 2]}, columns=['b'], dtype=dtype)
    if using_array_manager and dtype in tm.BYTES_DTYPES:
        td.mark_array_manager_not_yet_implemented(request)
    assert data.b.dtype.name == 'object'