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
def test_fwf_skip_blank_lines():
    data = '\n\nA         B            C            D\n\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n\n\n201162    502.953953   173.237159   12468.3\n\n'
    result = read_fwf(StringIO(data), skip_blank_lines=True)
    expected = DataFrame([[201158, 360.24294, 149.910199, 11950.7], [201159, 444.953632, 166.985655, 11788.4], [201162, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D'])
    tm.assert_frame_equal(result, expected)
    data = 'A         B            C            D\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n\n\n201162    502.953953   173.237159   12468.3\n'
    result = read_fwf(StringIO(data), skip_blank_lines=False)
    expected = DataFrame([[201158, 360.24294, 149.910199, 11950.7], [201159, 444.953632, 166.985655, 11788.4], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [201162, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D'])
    tm.assert_frame_equal(result, expected)