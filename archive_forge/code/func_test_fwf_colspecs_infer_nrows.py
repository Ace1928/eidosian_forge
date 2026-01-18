from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
@pytest.mark.parametrize('infer_nrows,exp_data', [(1, [[1, 2], [3, 8]]), (10, [[1, 2], [123, 98]])])
def test_fwf_colspecs_infer_nrows(infer_nrows, exp_data):
    data = '  1  2\n123 98\n'
    expected = DataFrame(exp_data)
    result = read_fwf(StringIO(data), infer_nrows=infer_nrows, header=None)
    tm.assert_frame_equal(result, expected)