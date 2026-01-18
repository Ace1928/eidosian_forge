import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_count_categorical(self):
    ser = Series(Categorical([np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True))
    result = ser.count()
    assert result == 2