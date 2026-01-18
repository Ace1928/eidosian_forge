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
def test_constructor_defaultdict(self, float_frame):
    data = {}
    float_frame.loc[:float_frame.index[10], 'B'] = np.nan
    for k, v in float_frame.items():
        dct = defaultdict(dict)
        dct.update(v.to_dict())
        data[k] = dct
    frame = DataFrame(data)
    expected = frame.reindex(index=float_frame.index)
    tm.assert_frame_equal(float_frame, expected)