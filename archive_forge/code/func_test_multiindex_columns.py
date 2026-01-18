from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multiindex_columns(df):
    cidx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df.columns = cidx
    expected = dedent('        \\begin{tabular}{lrrl}\n         & \\multicolumn{2}{r}{A} & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s = df.style.format(precision=2)
    assert expected == s.to_latex()
    expected = dedent('        \\begin{tabular}{lrrl}\n         & A & A & B \\\\\n         & a & b & c \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    s = df.style.format(precision=2)
    assert expected == s.to_latex(sparse_columns=False)