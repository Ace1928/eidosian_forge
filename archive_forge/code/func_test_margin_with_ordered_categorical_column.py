import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_margin_with_ordered_categorical_column(self):
    df = DataFrame({'First': ['B', 'B', 'C', 'A', 'B', 'C'], 'Second': ['C', 'B', 'B', 'B', 'C', 'A']})
    df['First'] = df['First'].astype(CategoricalDtype(ordered=True))
    customized_categories_order = ['C', 'A', 'B']
    df['First'] = df['First'].cat.reorder_categories(customized_categories_order)
    result = crosstab(df['First'], df['Second'], margins=True)
    expected_index = Index(['C', 'A', 'B', 'All'], name='First')
    expected_columns = Index(['A', 'B', 'C', 'All'], name='Second')
    expected_data = [[1, 1, 0, 2], [0, 1, 0, 1], [0, 1, 2, 3], [1, 3, 2, 6]]
    expected = DataFrame(expected_data, index=expected_index, columns=expected_columns)
    tm.assert_frame_equal(result, expected)