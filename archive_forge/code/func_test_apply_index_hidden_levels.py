from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_apply_index_hidden_levels():
    styler = DataFrame([[1]], index=MultiIndex.from_tuples([(0, 1)], names=['l0', 'l1']), columns=MultiIndex.from_tuples([(0, 1)], names=['c0', 'c1'])).style
    styler.hide(level=1)
    styler.map_index(lambda v: 'color: red;', level=0, axis=1)
    result = styler.to_latex(convert_css=True)
    expected = dedent('        \\begin{tabular}{lr}\n        c0 & \\color{red} 0 \\\\\n        c1 & 1 \\\\\n        l0 &  \\\\\n        0 & 1 \\\\\n        \\end{tabular}\n        ')
    assert result == expected