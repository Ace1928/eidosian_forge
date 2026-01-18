from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_deep_skip_rows(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n' + '\n'.join([','.join([str(i), str(i + 1), str(i + 2)]) for i in range(10)])
    condensed_data = 'a,b,c\n' + '\n'.join([','.join([str(i), str(i + 1), str(i + 2)]) for i in [0, 1, 2, 3, 4, 6, 8, 9]])
    result = parser.read_csv(StringIO(data), skiprows=[6, 8])
    condensed_result = parser.read_csv(StringIO(condensed_data))
    tm.assert_frame_equal(result, condensed_result)