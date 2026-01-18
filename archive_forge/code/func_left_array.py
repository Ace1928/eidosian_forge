import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture
def left_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype='boolean')