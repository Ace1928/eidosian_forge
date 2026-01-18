from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names, expected_names', [(['A', 'A'], ['A', 'A']), (['level_1', None], ['level_1', 'level_1'])])
@pytest.mark.parametrize('allow_duplicates', [False, True])
def test_column_name_duplicates(names, expected_names, allow_duplicates):
    s = Series([1], index=MultiIndex.from_arrays([[1], [1]], names=names))
    if allow_duplicates:
        result = s.reset_index(allow_duplicates=True)
        expected = DataFrame([[1, 1, 1]], columns=expected_names + [0])
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(ValueError, match='cannot insert'):
            s.reset_index()