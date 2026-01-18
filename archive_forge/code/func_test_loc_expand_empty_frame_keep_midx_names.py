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
def test_loc_expand_empty_frame_keep_midx_names(self):
    df = DataFrame(columns=['d'], index=MultiIndex.from_tuples([], names=['a', 'b', 'c']))
    df.loc[1, 2, 3] = 'foo'
    expected = DataFrame({'d': ['foo']}, index=MultiIndex.from_tuples([(1, 2, 3)], names=['a', 'b', 'c']))
    tm.assert_frame_equal(df, expected)