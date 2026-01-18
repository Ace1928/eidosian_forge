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
@pytest.mark.parametrize('lst', [[True, False, True], [True, True, True], [False, False, False]])
def test_getitem_boolean_list(self, lst):
    df = DataFrame(np.arange(12).reshape(3, 4))
    result = df[lst]
    expected = df.loc[df.index[lst]]
    tm.assert_frame_equal(result, expected)