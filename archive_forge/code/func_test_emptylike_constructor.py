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
@pytest.mark.parametrize('emptylike,expected_index,expected_columns', [([[]], RangeIndex(1), RangeIndex(0)), ([[], []], RangeIndex(2), RangeIndex(0)), ([(_ for _ in [])], RangeIndex(1), RangeIndex(0))])
def test_emptylike_constructor(self, emptylike, expected_index, expected_columns):
    expected = DataFrame(index=expected_index, columns=expected_columns)
    result = DataFrame(emptylike)
    tm.assert_frame_equal(result, expected)