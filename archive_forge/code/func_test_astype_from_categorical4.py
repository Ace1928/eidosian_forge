import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_from_categorical4(self):
    df = DataFrame({'cats': ['a', 'b', 'b', 'a', 'a', 'd'], 'vals': [1, 2, 3, 4, 5, 6]})
    cats = Categorical(['a', 'b', 'b', 'a', 'a', 'd'])
    exp_df = DataFrame({'cats': cats, 'vals': [1, 2, 3, 4, 5, 6]})
    df['cats'] = df['cats'].astype('category')
    tm.assert_frame_equal(exp_df, df)