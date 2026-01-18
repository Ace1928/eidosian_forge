from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
@pytest.mark.parametrize('series, plus', [(Series(1, index=[1, 2, 3]), False), (Series(1, index=list('ABC')), True), (Series(1, index=MultiIndex.from_product([range(3), range(3)])), False), (Series(1, index=MultiIndex.from_product([range(3), ['foo', 'bar']])), True)])
def test_info_memory_usage_qualified(series, plus):
    buf = StringIO()
    series.info(buf=buf)
    if plus:
        assert '+' in buf.getvalue()
    else:
        assert '+' not in buf.getvalue()