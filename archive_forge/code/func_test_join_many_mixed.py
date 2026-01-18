import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_many_mixed(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), columns=['A', 'B', 'C', 'D'])
    df['key'] = ['foo', 'bar'] * 4
    df1 = df.loc[:, ['A', 'B']]
    df2 = df.loc[:, ['C', 'D']]
    df3 = df.loc[:, ['key']]
    result = df1.join([df2, df3])
    tm.assert_frame_equal(result, df)