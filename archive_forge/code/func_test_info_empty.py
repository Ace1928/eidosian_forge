from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_empty():
    df = DataFrame()
    buf = StringIO()
    df.info(buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent("        <class 'pandas.core.frame.DataFrame'>\n        RangeIndex: 0 entries\n        Empty DataFrame\n")
    assert result == expected