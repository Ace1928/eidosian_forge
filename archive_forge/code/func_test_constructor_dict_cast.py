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
def test_constructor_dict_cast(self, using_infer_string):
    test_data = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
    frame = DataFrame(test_data, dtype=float)
    assert len(frame) == 3
    assert frame['B'].dtype == np.float64
    assert frame['A'].dtype == np.float64
    frame = DataFrame(test_data)
    assert len(frame) == 3
    assert frame['B'].dtype == np.object_ if not using_infer_string else 'string'
    assert frame['A'].dtype == np.float64