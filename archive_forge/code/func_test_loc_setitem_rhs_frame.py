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
@pytest.mark.parametrize('val, idxr', [('x', 'a'), ('x', ['a']), (1, 'a'), (1, ['a'])])
def test_loc_setitem_rhs_frame(self, idxr, val):
    df = DataFrame({'a': [1, 2]})
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.loc[:, idxr] = DataFrame({'a': [val, 11]}, index=[1, 2])
    expected = DataFrame({'a': [np.nan, val]})
    tm.assert_frame_equal(df, expected)