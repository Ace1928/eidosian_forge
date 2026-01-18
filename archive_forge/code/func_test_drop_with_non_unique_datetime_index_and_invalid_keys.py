import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_with_non_unique_datetime_index_and_invalid_keys():
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'], index=pd.date_range('2012', freq='h', periods=5))
    df = df.iloc[[0, 2, 2, 3]].copy()
    with pytest.raises(KeyError, match='not found in axis'):
        df.drop(['a', 'b'])