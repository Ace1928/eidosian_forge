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
def test_names_and_infer_colspecs():
    data = 'X   Y   Z\n      959.0    345   22.2\n    '
    result = read_fwf(StringIO(data), skiprows=1, usecols=[0, 2], names=['a', 'b'])
    expected = DataFrame({'a': [959.0], 'b': 22.2})
    tm.assert_frame_equal(result, expected)