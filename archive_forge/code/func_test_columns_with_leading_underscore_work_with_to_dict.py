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
def test_columns_with_leading_underscore_work_with_to_dict(self):
    col_underscore = '_b'
    df = DataFrame({'a': [1, 2], col_underscore: [3, 4]})
    d = df.to_dict(orient='records')
    ref_d = [{'a': 1, col_underscore: 3}, {'a': 2, col_underscore: 4}]
    assert ref_d == d