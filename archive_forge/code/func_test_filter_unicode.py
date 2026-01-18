import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('name,expected', [('a', DataFrame({'a': [1, 2]})), ('a', DataFrame({'a': [1, 2]})), ('あ', DataFrame({'あ': [3, 4]}))])
def test_filter_unicode(self, name, expected):
    df = DataFrame({'a': [1, 2], 'あ': [3, 4]})
    tm.assert_frame_equal(df.filter(like=name), expected)
    tm.assert_frame_equal(df.filter(regex=name), expected)