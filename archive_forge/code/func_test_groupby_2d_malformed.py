from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_2d_malformed():
    d = DataFrame(index=range(2))
    d['group'] = ['g1', 'g2']
    d['zeros'] = [0, 0]
    d['ones'] = [1, 1]
    d['label'] = ['l1', 'l2']
    tmp = d.groupby(['group']).mean(numeric_only=True)
    res_values = np.array([[0.0, 1.0], [0.0, 1.0]])
    tm.assert_index_equal(tmp.columns, Index(['zeros', 'ones']))
    tm.assert_numpy_array_equal(tmp.values, res_values)