import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_astype_dup_col(self):
    df = DataFrame([{'a': 'b'}])
    df = concat([df, df], axis=1)
    result = df.astype('category')
    expected = DataFrame(np.array(['b', 'b']).reshape(1, 2), columns=['a', 'a']).astype('category')
    tm.assert_frame_equal(result, expected)