from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
def test_info_wide():
    s = Series(np.random.randn(101))
    msg = 'Argument `max_cols` can only be passed in DataFrame.info, not Series.info'
    with pytest.raises(ValueError, match=msg):
        s.info(max_cols=1)