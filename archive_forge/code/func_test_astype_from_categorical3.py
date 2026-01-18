import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_from_categorical3(self):
    df = DataFrame({'cats': [1, 2, 3, 4, 5, 6], 'vals': [1, 2, 3, 4, 5, 6]})
    cats = Categorical([1, 2, 3, 4, 5, 6])
    exp_df = DataFrame({'cats': cats, 'vals': [1, 2, 3, 4, 5, 6]})
    df['cats'] = df['cats'].astype('category')
    tm.assert_frame_equal(exp_df, df)