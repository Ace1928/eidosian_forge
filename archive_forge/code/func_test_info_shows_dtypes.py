from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
def test_info_shows_dtypes():
    dtypes = ['int64', 'float64', 'datetime64[ns]', 'timedelta64[ns]', 'complex128', 'object', 'bool']
    n = 10
    for dtype in dtypes:
        s = Series(np.random.randint(2, size=n).astype(dtype))
        buf = StringIO()
        s.info(buf=buf)
        res = buf.getvalue()
        name = f'{n:d} non-null     {dtype}'
        assert name in res