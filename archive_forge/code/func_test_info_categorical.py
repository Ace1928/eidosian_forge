from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
def test_info_categorical():
    idx = CategoricalIndex(['a', 'b'])
    s = Series(np.zeros(2), index=idx)
    buf = StringIO()
    s.info(buf=buf)