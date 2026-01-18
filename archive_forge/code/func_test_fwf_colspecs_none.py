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
@pytest.mark.parametrize('colspecs,exp_data', [([(0, 3), (3, None)], [[123, 456], [456, 789]]), ([(None, 3), (3, 6)], [[123, 456], [456, 789]]), ([(0, None), (3, None)], [[123456, 456], [456789, 789]]), ([(None, None), (3, 6)], [[123456, 456], [456789, 789]])])
def test_fwf_colspecs_none(colspecs, exp_data):
    data = '123456\n456789\n'
    expected = DataFrame(exp_data)
    result = read_fwf(StringIO(data), colspecs=colspecs, header=None)
    tm.assert_frame_equal(result, expected)