import numpy as np
import pytest
from pandas import (
@pytest.fixture
def float_string_frame():
    """
    Fixture for DataFrame of floats and strings with index of unique strings

    Columns are ['A', 'B', 'C', 'D', 'foo'].
    """
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), index=Index([f'foo_{i}' for i in range(30)], dtype=object), columns=Index(list('ABCD'), dtype=object))
    df['foo'] = 'bar'
    return df