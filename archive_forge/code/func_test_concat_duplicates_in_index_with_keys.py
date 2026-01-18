import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_duplicates_in_index_with_keys(self):
    index = [1, 1, 3]
    data = [1, 2, 3]
    df = DataFrame(data=data, index=index)
    result = concat([df], keys=['A'], names=['ID', 'date'])
    mi = pd.MultiIndex.from_product([['A'], index], names=['ID', 'date'])
    expected = DataFrame(data=data, index=mi)
    tm.assert_frame_equal(result, expected)
    tm.assert_index_equal(result.index.levels[1], Index([1, 3], name='date'))