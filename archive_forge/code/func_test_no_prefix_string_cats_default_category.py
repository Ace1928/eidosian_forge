import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('default_category, expected', [pytest.param('c', DataFrame({'': ['a', 'b', 'c']}), id='default_category is a str'), pytest.param(1, DataFrame({'': ['a', 'b', 1]}), id='default_category is a int'), pytest.param(1.25, DataFrame({'': ['a', 'b', 1.25]}), id='default_category is a float'), pytest.param(0, DataFrame({'': ['a', 'b', 0]}), id='default_category is a 0'), pytest.param(False, DataFrame({'': ['a', 'b', False]}), id='default_category is a bool'), pytest.param((1, 2), DataFrame({'': ['a', 'b', (1, 2)]}), id='default_category is a tuple')])
def test_no_prefix_string_cats_default_category(default_category, expected, using_infer_string):
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0]})
    result = from_dummies(dummies, default_category=default_category)
    if using_infer_string:
        expected[''] = expected[''].astype('string[pyarrow_numpy]')
    tm.assert_frame_equal(result, expected)