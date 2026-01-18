from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('levels', [[['x', 'y']], [['x', 'y', 'y']]])
def test_concat_with_levels_with_none_keys(self, levels):
    df1 = DataFrame({'A': [1]}, index=['x'])
    df2 = DataFrame({'A': [1]}, index=['y'])
    msg = 'levels supported only when keys is not None'
    with pytest.raises(ValueError, match=msg):
        concat([df1, df2], levels=levels)