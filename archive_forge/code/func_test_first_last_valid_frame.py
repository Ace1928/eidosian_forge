import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('data,idx,expected_first,expected_last', [({'A': [1, 2, 3]}, [1, 1, 2], 1, 2), ({'A': [1, 2, 3]}, [1, 2, 2], 1, 2), ({'A': [1, 2, 3, 4]}, ['d', 'd', 'd', 'd'], 'd', 'd'), ({'A': [1, np.nan, 3]}, [1, 1, 2], 1, 2), ({'A': [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2), ({'A': [1, np.nan, 3]}, [1, 2, 2], 1, 2)])
def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
    df = DataFrame(data, index=idx)
    assert expected_first == df.first_valid_index()
    assert expected_last == df.last_valid_index()