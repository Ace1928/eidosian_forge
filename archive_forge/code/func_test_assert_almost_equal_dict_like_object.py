import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    dict_val = 1
    real_dict = {'a': val}

    class DictLikeObj:

        def keys(self):
            return ('a',)

        def __getitem__(self, item):
            if item == 'a':
                return dict_val
    func = _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    func(real_dict, DictLikeObj(), check_dtype=False)