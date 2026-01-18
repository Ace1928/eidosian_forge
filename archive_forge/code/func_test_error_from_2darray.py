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
@pytest.mark.parametrize('col_a, col_b', [([[1], [2]], np.array([[1], [2]])), (np.array([[1], [2]]), [[1], [2]]), (np.array([[1], [2]]), np.array([[1], [2]]))])
def test_error_from_2darray(self, col_a, col_b):
    msg = 'Per-column arrays must each be 1-dimensional'
    with pytest.raises(ValueError, match=msg):
        DataFrame({'a': col_a, 'b': col_b})