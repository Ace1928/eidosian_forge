import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_duplicate_names(self):
    s1 = Series(range(3), name='foo')
    s2_foo = Series(range(1, 4), name='foo')
    s2_bar = Series(range(1, 4), name='bar')
    s3 = Series(range(3), name='waldo')
    mapper = {'bar': 'foo'}
    result = crosstab(s1, s2_foo)
    expected = crosstab(s1, s2_bar).rename_axis(columns=mapper, axis=1)
    tm.assert_frame_equal(result, expected)
    result = crosstab([s1, s2_foo], s3)
    expected = crosstab([s1, s2_bar], s3).rename_axis(index=mapper, axis=0)
    tm.assert_frame_equal(result, expected)
    result = crosstab(s3, [s1, s2_foo])
    expected = crosstab(s3, [s1, s2_bar]).rename_axis(columns=mapper, axis=1)
    tm.assert_frame_equal(result, expected)