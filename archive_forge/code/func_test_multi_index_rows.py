import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_multi_index_rows():
    df = pd.DataFrame({'A': np.array([[0, 1, 2], np.nan, [], (3, 4)], dtype=object), 'B': 1}, index=pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]))
    result = df.explode('A')
    expected = pd.DataFrame({'A': pd.Series([0, 1, 2, np.nan, np.nan, 3, 4], index=pd.MultiIndex.from_tuples([('a', 1), ('a', 1), ('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 2)]), dtype=object), 'B': 1})
    tm.assert_frame_equal(result, expected)