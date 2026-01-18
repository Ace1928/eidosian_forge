from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_with_duplicated_levels(self):
    df1 = DataFrame({'A': [1]}, index=['x'])
    df2 = DataFrame({'A': [1]}, index=['y'])
    msg = "Level values not unique: \\['x', 'y', 'y'\\]"
    with pytest.raises(ValueError, match=msg):
        concat([df1, df2], keys=['x', 'y'], levels=[['x', 'y', 'y']])