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
def test_constructor_maskedarray(self):
    self._check_basic_constructor(ma.masked_all)
    mat = ma.masked_all((2, 3), dtype=float)
    mat[0, 0] = 1.0
    mat[1, 2] = 2.0
    frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
    assert 1.0 == frame['A'][1]
    assert 2.0 == frame['C'][2]
    mat = ma.masked_all((2, 3), dtype=float)
    frame = DataFrame(mat, columns=['A', 'B', 'C'], index=[1, 2])
    assert np.all(~np.asarray(frame == frame))