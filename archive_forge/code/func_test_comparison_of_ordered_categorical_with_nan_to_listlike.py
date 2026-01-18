import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_comparison_of_ordered_categorical_with_nan_to_listlike(self, compare_operators_no_eq_ne):
    cat = Categorical([1, 2, 3, None], categories=[1, 2, 3], ordered=True)
    other = Categorical([2, 2, 2, 2], categories=[1, 2, 3], ordered=True)
    expected = getattr(np.array(cat), compare_operators_no_eq_ne)(2)
    actual = getattr(cat, compare_operators_no_eq_ne)(other)
    tm.assert_numpy_array_equal(actual, expected)