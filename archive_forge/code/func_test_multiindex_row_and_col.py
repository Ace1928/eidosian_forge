from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_multiindex_row_and_col(df_ext):
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & \\multicolumn{2}{l}{Z} & Y \\\\\n         &  & a & b & c \\\\\n        \\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\\n         & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    styler = df_ext.style.format(precision=2)
    result = styler.to_latex(multirow_align='b', multicol_align='l')
    assert result == expected
    expected = dedent('        \\begin{tabular}{llrrl}\n         &  & Z & Z & Y \\\\\n         &  & a & b & c \\\\\n        A & a & 0 & -0.61 & ab \\\\\n        A & b & 1 & -1.22 & cd \\\\\n        B & c & 2 & -2.22 & de \\\\\n        \\end{tabular}\n        ')
    result = styler.to_latex(sparse_index=False, sparse_columns=False)
    assert result == expected