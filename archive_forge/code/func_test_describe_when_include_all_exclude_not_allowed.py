import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('exclude', ['x', 'y', ['x', 'y'], ['x', 'z']])
def test_describe_when_include_all_exclude_not_allowed(self, exclude):
    """
        When include is 'all', then setting exclude != None is not allowed.
        """
    df = DataFrame({'x': [1], 'y': [2], 'z': [3]})
    msg = "exclude must be None when include is 'all'"
    with pytest.raises(ValueError, match=msg):
        df.describe(include='all', exclude=exclude)