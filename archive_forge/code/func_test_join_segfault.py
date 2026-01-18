from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_segfault(self):
    df1 = DataFrame({'a': [1, 1], 'b': [1, 2], 'x': [1, 2]})
    df2 = DataFrame({'a': [2, 2], 'b': [1, 2], 'y': [1, 2]})
    df1 = df1.set_index(['a', 'b'])
    df2 = df2.set_index(['a', 'b'])
    for how in ['left', 'right', 'outer']:
        df1.join(df2, how=how)