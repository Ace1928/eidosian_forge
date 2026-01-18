from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_iloc_setitem_enlarge_no_warning(self, warn_copy_on_write):
    df = DataFrame(columns=['a', 'b'])
    expected = df.copy()
    view = df[:]
    df.iloc[:, 0] = np.array([1, 2], dtype=np.float64)
    tm.assert_frame_equal(view, expected)