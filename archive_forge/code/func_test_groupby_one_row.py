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
def test_groupby_one_row():
    msg = "^'Z'$"
    df1 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), columns=list('ABCD'))
    with pytest.raises(KeyError, match=msg):
        df1.groupby('Z')
    df2 = DataFrame(np.random.default_rng(2).standard_normal((2, 4)), columns=list('ABCD'))
    with pytest.raises(KeyError, match=msg):
        df2.groupby('Z')