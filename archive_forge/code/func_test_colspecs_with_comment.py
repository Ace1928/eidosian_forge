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
def test_colspecs_with_comment():
    result = read_fwf(StringIO('#\nA1K\n'), colspecs=[(1, 2), (2, 3)], comment='#', header=None)
    expected = DataFrame([[1, 'K']], columns=[0, 1])
    tm.assert_frame_equal(result, expected)