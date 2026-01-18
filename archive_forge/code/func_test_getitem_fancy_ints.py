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
def test_getitem_fancy_ints(self, float_frame):
    result = float_frame.iloc[[1, 4, 7]]
    expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
    tm.assert_frame_equal(result, expected)
    result = float_frame.iloc[:, [2, 0, 1]]
    expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
    tm.assert_frame_equal(result, expected)