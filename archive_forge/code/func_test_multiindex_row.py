from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multiindex_row(df_ext):
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index = ridx
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex()
    assert expected == result
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False)
    assert expected == result