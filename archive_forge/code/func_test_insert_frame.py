import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_frame(self):
    df = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    msg = 'Expected a one-dimensional object, got a DataFrame with 2 columns instead.'
    with pytest.raises(ValueError, match=msg):
        df.insert(1, 'newcol', df)