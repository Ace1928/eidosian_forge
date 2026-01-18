from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
@pytest.mark.xfail(not PYPY, reason='on PyPy deep=True does not change result')
def test_info_memory_usage_deep_pypy():
    s_with_object_index = Series({'a': [1]}, index=['foo'])
    assert s_with_object_index.memory_usage(index=True, deep=True) == s_with_object_index.memory_usage(index=True)
    s_object = Series({'a': ['a']})
    assert s_object.memory_usage(deep=True) == s_object.memory_usage()