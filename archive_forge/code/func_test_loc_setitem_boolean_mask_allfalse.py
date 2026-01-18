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
def test_loc_setitem_boolean_mask_allfalse(self):
    df = DataFrame({'a': ['1', '2', '3'], 'b': ['11', '22', '33'], 'c': ['111', '222', '333']})
    result = df.copy()
    result.loc[result.b.isna(), 'a'] = result.a.copy()
    tm.assert_frame_equal(result, df)