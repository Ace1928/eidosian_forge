import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_last_nth_dtypes2():
    idx = list(range(10))
    idx.append(9)
    ser = Series(data=range(11), index=idx, name='IntCol')
    assert ser.dtype == 'int64'
    f = ser.groupby(level=0).first()
    assert f.dtype == 'int64'