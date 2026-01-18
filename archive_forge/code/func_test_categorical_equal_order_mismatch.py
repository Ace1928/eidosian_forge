import pytest
from pandas import Categorical
import pandas._testing as tm
@pytest.mark.parametrize('check_category_order', [True, False])
def test_categorical_equal_order_mismatch(check_category_order):
    c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 4], categories=[4, 3, 2, 1])
    kwargs = {'check_category_order': check_category_order}
    if check_category_order:
        msg = "Categorical\\.categories are different\n\nCategorical\\.categories values are different \\(100\\.0 %\\)\n\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)\n\\[right\\]: Index\\(\\[4, 3, 2, 1\\], dtype='int64'\\)"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_categorical_equal(c1, c2, **kwargs)
    else:
        tm.assert_categorical_equal(c1, c2, **kwargs)