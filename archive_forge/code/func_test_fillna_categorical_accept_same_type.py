from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('input, input_fillna, expected_data, expected_categories', [(['A', 'B', None, 'A'], 'B', ['A', 'B', 'B', 'A'], ['A', 'B']), (['A', 'B', np.nan, 'A'], 'B', ['A', 'B', 'B', 'A'], ['A', 'B'])])
def test_fillna_categorical_accept_same_type(self, input, input_fillna, expected_data, expected_categories):
    cat = Categorical(input)
    ser = Series(cat).fillna(input_fillna)
    filled = cat.fillna(ser)
    result = cat.fillna(filled)
    expected = Categorical(expected_data, categories=expected_categories)
    tm.assert_categorical_equal(result, expected)