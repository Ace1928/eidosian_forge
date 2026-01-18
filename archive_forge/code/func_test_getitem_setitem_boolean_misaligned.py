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
def test_getitem_setitem_boolean_misaligned(self, float_frame):
    mask = float_frame['A'][::-1] > 1
    result = float_frame.loc[mask]
    expected = float_frame.loc[mask[::-1]]
    tm.assert_frame_equal(result, expected)
    cp = float_frame.copy()
    expected = float_frame.copy()
    cp.loc[mask] = 0
    expected.loc[mask] = 0
    tm.assert_frame_equal(cp, expected)