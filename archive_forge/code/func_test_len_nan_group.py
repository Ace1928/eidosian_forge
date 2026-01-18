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
def test_len_nan_group():
    df = DataFrame({'a': [np.nan] * 3, 'b': [1, 2, 3]})
    assert len(df.groupby('a')) == 0
    assert len(df.groupby('b')) == 3
    assert len(df.groupby(['a', 'b'])) == 3