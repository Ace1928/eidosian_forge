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
@pytest.mark.parametrize('size, header_exp, separator_exp, first_line_exp, last_line_exp', [(4, ' #   Column  Non-Null Count  Dtype  ', '---  ------  --------------  -----  ', ' 0   0       3 non-null      float64', ' 3   3       3 non-null      float64'), (11, ' #   Column  Non-Null Count  Dtype  ', '---  ------  --------------  -----  ', ' 0   0       3 non-null      float64', ' 10  10      3 non-null      float64'), (101, ' #    Column  Non-Null Count  Dtype  ', '---   ------  --------------  -----  ', ' 0    0       3 non-null      float64', ' 100  100     3 non-null      float64'), (1001, ' #     Column  Non-Null Count  Dtype  ', '---    ------  --------------  -----  ', ' 0     0       3 non-null      float64', ' 1000  1000    3 non-null      float64'), (10001, ' #      Column  Non-Null Count  Dtype  ', '---     ------  --------------  -----  ', ' 0      0       3 non-null      float64', ' 10000  10000   3 non-null      float64')])
def test_info_verbose_with_counts_spacing(size, header_exp, separator_exp, first_line_exp, last_line_exp):
    """Test header column, spacer, first line and last line in verbose mode."""
    frame = DataFrame(np.random.randn(3, size))
    with StringIO() as buf:
        frame.info(verbose=True, show_counts=True, buf=buf)
        all_lines = buf.getvalue().splitlines()
    table = all_lines[3:-2]
    header, separator, first_line, *rest, last_line = table
    assert header == header_exp
    assert separator == separator_exp
    assert first_line == first_line_exp
    assert last_line == last_line_exp