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
def test_loc_rhs_empty_warning(self):
    df = DataFrame(columns=['a', 'b'])
    expected = df.copy()
    rhs = DataFrame(columns=['a'])
    with tm.assert_produces_warning(None):
        df.loc[:, 'a'] = rhs
    tm.assert_frame_equal(df, expected)