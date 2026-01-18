import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def cat_series(self, dtype, ordered):
    cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))
    input2 = np.array([1, 2, 3, 5, 3, 2, 4], dtype=np.dtype(dtype))
    cat = Categorical(input2, categories=cat_array, ordered=ordered)
    tc2 = Series(cat)
    return tc2