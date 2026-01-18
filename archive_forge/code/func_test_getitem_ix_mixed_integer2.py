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
def test_getitem_ix_mixed_integer2(self):
    df = DataFrame({'rna': (1.5, 2.2, 3.2, 4.5), -1000: [11, 21, 36, 40], 0: [10, 22, 43, 34], 1000: [0, 10, 20, 30]}, columns=['rna', -1000, 0, 1000])
    result = df[[1000]]
    expected = df.iloc[:, [3]]
    tm.assert_frame_equal(result, expected)
    result = df[[-1000]]
    expected = df.iloc[:, [1]]
    tm.assert_frame_equal(result, expected)