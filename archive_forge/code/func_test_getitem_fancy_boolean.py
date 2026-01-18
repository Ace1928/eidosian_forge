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
def test_getitem_fancy_boolean(self, float_frame):
    f = float_frame
    ix = f.loc
    expected = f.reindex(columns=['B', 'D'])
    result = ix[:, [False, True, False, True]]
    tm.assert_frame_equal(result, expected)
    expected = f.reindex(index=f.index[5:10], columns=['B', 'D'])
    result = ix[f.index[5:10], [False, True, False, True]]
    tm.assert_frame_equal(result, expected)
    boolvec = f.index > f.index[7]
    expected = f.reindex(index=f.index[boolvec])
    result = ix[boolvec]
    tm.assert_frame_equal(result, expected)
    result = ix[boolvec, :]
    tm.assert_frame_equal(result, expected)
    result = ix[boolvec, f.columns[2:]]
    expected = f.reindex(index=f.index[boolvec], columns=['C', 'D'])
    tm.assert_frame_equal(result, expected)