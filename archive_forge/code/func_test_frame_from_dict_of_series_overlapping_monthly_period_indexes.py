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
def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self):
    rng1 = pd.period_range('1/1/1999', '1/1/2012', freq='M')
    s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)
    rng2 = pd.period_range('1/1/1980', '12/1/2001', freq='M')
    s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
    df = DataFrame({'s1': s1, 's2': s2})
    exp = pd.period_range('1/1/1980', '1/1/2012', freq='M')
    tm.assert_index_equal(df.index, exp)