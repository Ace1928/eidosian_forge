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
@pytest.mark.parametrize('fixture_func_name', ['int_frame', 'float_frame', 'datetime_frame', 'duplicate_columns_frame'])
def test_info_smoke_test(fixture_func_name, request):
    frame = request.getfixturevalue(fixture_func_name)
    buf = StringIO()
    frame.info(buf=buf)
    result = buf.getvalue().splitlines()
    assert len(result) > 10