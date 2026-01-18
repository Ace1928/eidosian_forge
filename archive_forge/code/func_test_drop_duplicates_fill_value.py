import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_drop_duplicates_fill_value():
    df = pd.DataFrame(np.zeros((5, 5))).apply(lambda x: SparseArray(x, fill_value=0))
    result = df.drop_duplicates()
    expected = pd.DataFrame({i: SparseArray([0.0], fill_value=0) for i in range(5)})
    tm.assert_frame_equal(result, expected)