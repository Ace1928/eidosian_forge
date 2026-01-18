import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_preserve_index_name(self):
    df1 = DataFrame(columns=['A', 'B', 'C'])
    df1 = df1.set_index(['A'])
    df2 = DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=['A', 'B', 'C'])
    df2 = df2.set_index(['A'])
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df1._append(df2)
    assert result.index.name == 'A'