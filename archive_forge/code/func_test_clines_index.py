from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('clines, exp', [('all;index', '\n\\cline{1-1}'), ('all;data', '\n\\cline{1-2}'), ('skip-last;index', ''), ('skip-last;data', ''), (None, '')])
@pytest.mark.parametrize('env', ['table', 'longtable'])
def test_clines_index(clines, exp, env):
    df = DataFrame([[1], [2], [3], [4]])
    result = df.style.to_latex(clines=clines, environment=env)
    expected = f'0 & 1 \\\\{exp}\n1 & 2 \\\\{exp}\n2 & 3 \\\\{exp}\n3 & 4 \\\\{exp}\n'
    assert expected in result