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
def test_construct_with_two_categoricalindex_series(self):
    s1 = Series([39, 6, 4], index=CategoricalIndex(['female', 'male', 'unknown']))
    s2 = Series([2, 152, 2, 242, 150], index=CategoricalIndex(['f', 'female', 'm', 'male', 'unknown']))
    result = DataFrame([s1, s2])
    expected = DataFrame(np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]), columns=['female', 'male', 'unknown', 'f', 'm'])
    tm.assert_frame_equal(result, expected)