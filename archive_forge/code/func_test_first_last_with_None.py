import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_None(method):
    df = DataFrame.from_dict({'id': ['a'], 'value': [None]})
    groups = df.groupby('id', as_index=False)
    result = getattr(groups, method)()
    tm.assert_frame_equal(result, df)