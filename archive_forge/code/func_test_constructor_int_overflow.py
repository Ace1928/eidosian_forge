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
@pytest.mark.parametrize('values', [np.array([2 ** 64], dtype=object), np.array([2 ** 65]), [2 ** 64 + 1], np.array([-2 ** 63 - 4], dtype=object), np.array([-2 ** 64 - 1]), [-2 ** 65 - 2]])
def test_constructor_int_overflow(self, values):
    value = values[0]
    result = DataFrame(values)
    assert result[0].dtype == object
    assert result[0][0] == value