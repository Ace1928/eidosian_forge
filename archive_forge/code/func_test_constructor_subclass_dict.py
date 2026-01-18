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
def test_constructor_subclass_dict(self, dict_subclass):
    data = {'col1': dict_subclass(((x, 10.0 * x) for x in range(10))), 'col2': dict_subclass(((x, 20.0 * x) for x in range(10)))}
    df = DataFrame(data)
    refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
    tm.assert_frame_equal(refdf, df)
    data = dict_subclass(data.items())
    df = DataFrame(data)
    tm.assert_frame_equal(refdf, df)