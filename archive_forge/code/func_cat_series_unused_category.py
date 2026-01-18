import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def cat_series_unused_category(self, dtype, ordered):
    cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))
    input1 = np.array([1, 2, 3, 3], dtype=np.dtype(dtype))
    cat = Categorical(input1, categories=cat_array, ordered=ordered)
    tc1 = Series(cat)
    return tc1