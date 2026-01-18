import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_cumulative_ops_smoke(self):
    df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
    df.cummax()
    df.cummin()
    df.cumsum()
    dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
    dm.cumsum()