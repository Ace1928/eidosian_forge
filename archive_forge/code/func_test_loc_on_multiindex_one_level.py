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
def test_loc_on_multiindex_one_level(self):
    df = DataFrame(data=[[0], [1]], index=MultiIndex.from_tuples([('a',), ('b',)], names=['first']))
    expected = DataFrame(data=[[0]], index=MultiIndex.from_tuples([('a',)], names=['first']))
    result = df.loc['a']
    tm.assert_frame_equal(result, expected)