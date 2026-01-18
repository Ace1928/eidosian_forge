from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_hide_index_latex(styler):
    styler.hide([0], axis=0)
    result = styler.to_latex()
    expected = dedent('    \\begin{tabular}{lrrl}\n     & A & B & C \\\\\n    1 & 1 & -1.22 & cd \\\\\n    \\end{tabular}\n    ')
    assert expected == result