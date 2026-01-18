import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_iloc_integer_locations():
    data = [['str00', 'str01'], ['str10', 'str11'], ['str20', 'srt21'], ['str30', 'str31'], ['str40', 'str41']]
    index = MultiIndex.from_tuples([('CC', 'A'), ('CC', 'B'), ('CC', 'B'), ('BB', 'a'), ('BB', 'b')])
    expected = DataFrame(data)
    df = DataFrame(data, index=index)
    result = DataFrame([[df.iloc[r, c] for c in range(2)] for r in range(5)])
    tm.assert_frame_equal(result, expected)