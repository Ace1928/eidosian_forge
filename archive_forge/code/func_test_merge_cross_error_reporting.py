import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('kwargs', [{'left_index': True}, {'right_index': True}, {'on': 'a'}, {'left_on': 'a'}, {'right_on': 'b'}])
def test_merge_cross_error_reporting(kwargs):
    left = DataFrame({'a': [1, 3]})
    right = DataFrame({'b': [3, 4]})
    msg = 'Can not pass on, right_on, left_on or set right_index=True or left_index=True'
    with pytest.raises(MergeError, match=msg):
        merge(left, right, how='cross', **kwargs)