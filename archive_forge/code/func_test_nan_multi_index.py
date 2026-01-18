from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_nan_multi_index(all_parsers):
    parser = all_parsers
    data = 'A,B,B\nX,Y,Z\n1,2,inf'
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=list(range(2)), na_values={('B', 'Z'): 'inf'})
        return
    result = parser.read_csv(StringIO(data), header=list(range(2)), na_values={('B', 'Z'): 'inf'})
    expected = DataFrame({('A', 'X'): [1], ('B', 'Y'): [2], ('B', 'Z'): [np.nan]})
    tm.assert_frame_equal(result, expected)