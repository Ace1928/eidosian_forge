import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_expansion_label_unused_category(self, df2):
    df = df2.copy()
    df.loc['e'] = 20
    result = df.loc[['a', 'b', 'e']]
    exp_index = CategoricalIndex(list('aaabbe'), categories=list('cabe'), name='B')
    expected = DataFrame({'A': [0, 1, 5, 2, 3, 20]}, index=exp_index)
    tm.assert_frame_equal(result, expected)