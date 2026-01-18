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
def test_messed_up_data():
    test = '\n   Account          Name             Balance     Credit Limit   Account Created\n       101                           10000.00                       1/17/1998\n       312     Gerard Butler         90.00       1000.00\n\n       761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006\n  317          Bill Murray           789.65\n'.strip('\r\n')
    colspecs = ((2, 10), (15, 33), (37, 45), (49, 61), (64, 79))
    expected = read_fwf(StringIO(test), colspecs=colspecs)
    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)