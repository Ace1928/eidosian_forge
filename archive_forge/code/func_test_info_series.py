from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
@pytest.mark.parametrize('verbose', [True, False])
def test_info_series(lexsorted_two_level_string_multiindex, verbose):
    index = lexsorted_two_level_string_multiindex
    ser = Series(range(len(index)), index=index, name='sth')
    buf = StringIO()
    ser.info(verbose=verbose, buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent("        <class 'pandas.core.series.Series'>\n        MultiIndex: 10 entries, ('foo', 'one') to ('qux', 'three')\n        ")
    if verbose:
        expected += textwrap.dedent('            Series name: sth\n            Non-Null Count  Dtype\n            --------------  -----\n            10 non-null     int64\n            ')
    expected += textwrap.dedent(f'        dtypes: int64(1)\n        memory usage: {ser.memory_usage()}.0+ bytes\n        ')
    assert result == expected