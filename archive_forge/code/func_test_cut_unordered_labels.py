import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('data, bins, labels, expected_codes, expected_labels', [([15, 17, 19], [14, 16, 18, 20], ['C', 'B', 'A'], [0, 1, 2], ['C', 'B', 'A']), ([1, 3, 5], [0, 2, 4, 6, 8], [3, 0, 1, 2], [0, 1, 2], [3, 0, 1, 2])])
def test_cut_unordered_labels(data, bins, labels, expected_codes, expected_labels):
    result = cut(data, bins=bins, labels=labels, ordered=False)
    expected = Categorical.from_codes(expected_codes, categories=expected_labels, ordered=False)
    tm.assert_categorical_equal(result, expected)