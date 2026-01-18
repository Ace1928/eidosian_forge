from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multirow_naive(df_ext):
    ridx = MultiIndex.from_tuples([('X', 'x'), ('X', 'y'), ('Y', 'z')])
    df_ext.index = ridx
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & A & B & C \\\\\n        X & x & 0 & -0.61 & ab \\\\\n         & y & 1 & -1.22 & cd \\\\\n        Y & z & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align='naive')
    assert expected == result