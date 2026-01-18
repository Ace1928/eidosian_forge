from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_mixed_bug(self):
    idx = Index(['a', 'b', 'c', 'e'])
    ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
    ser2 = Series(['a', 'b', 'c', 'e'], index=idx)
    ser3 = Series([12, 4, 5, 97], index=idx)
    frame1 = DataFrame({'col0': ser1, 'col2': ser2, 'col3': ser3})
    idx = Index(['a', 'b', 'c', 'f'])
    ser1 = Series([5.0, -9.0, 4.0, 100.0], index=idx)
    ser2 = Series(['a', 'b', 'c', 'f'], index=idx)
    ser3 = Series([12, 4, 5, 97], index=idx)
    frame2 = DataFrame({'col1': ser1, 'col2': ser2, 'col5': ser3})
    combined = frame1.combine_first(frame2)
    assert len(combined.columns) == 5