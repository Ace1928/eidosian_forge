from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('clines, expected', [(None, dedent('            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            ')), ('skip-last;index', dedent('            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\cline{1-2}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            \\cline{1-2}\n            ')), ('skip-last;data', dedent('            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n             & Y & 2 \\\\\n            \\cline{1-3}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n             & Y & 4 \\\\\n            \\cline{1-3}\n            ')), ('all;index', dedent('            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n            \\cline{2-2}\n             & Y & 2 \\\\\n            \\cline{1-2} \\cline{2-2}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n            \\cline{2-2}\n             & Y & 4 \\\\\n            \\cline{1-2} \\cline{2-2}\n            ')), ('all;data', dedent('            \\multirow[c]{2}{*}{A} & X & 1 \\\\\n            \\cline{2-3}\n             & Y & 2 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\multirow[c]{2}{*}{B} & X & 3 \\\\\n            \\cline{2-3}\n             & Y & 4 \\\\\n            \\cline{1-3} \\cline{2-3}\n            '))])
@pytest.mark.parametrize('env', ['table'])
def test_clines_multiindex(clines, expected, env):
    midx = MultiIndex.from_product([['A', '-', 'B'], [0], ['X', 'Y']])
    df = DataFrame([[1], [2], [99], [99], [3], [4]], index=midx)
    styler = df.style
    styler.hide([('-', 0, 'X'), ('-', 0, 'Y')])
    styler.hide(level=1)
    result = styler.to_latex(clines=clines, environment=env)
    assert expected in result