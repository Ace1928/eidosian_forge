import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('get_dummies_kwargs,expected', [({'data': DataFrame({'ä': ['a']})}, DataFrame({'ä_a': [True]})), ({'data': DataFrame({'x': ['ä']})}, DataFrame({'x_ä': [True]})), ({'data': DataFrame({'x': ['a']}), 'prefix': 'ä'}, DataFrame({'ä_a': [True]})), ({'data': DataFrame({'x': ['a']}), 'prefix_sep': 'ä'}, DataFrame({'xäa': [True]}))])
def test_dataframe_dummies_unicode(self, get_dummies_kwargs, expected):
    result = get_dummies(**get_dummies_kwargs)
    tm.assert_frame_equal(result, expected)