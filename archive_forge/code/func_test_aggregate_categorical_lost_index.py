import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', ['min', 'max'])
def test_aggregate_categorical_lost_index(func: str):
    ds = Series(['b'], dtype='category').cat.as_ordered()
    df = DataFrame({'A': [1997], 'B': ds})
    result = df.groupby('A').agg({'B': func})
    expected = DataFrame({'B': ['b']}, index=Index([1997], name='A'))
    expected['B'] = expected['B'].astype(ds.dtype)
    tm.assert_frame_equal(result, expected)