from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('multicol_align, siunitx, header', [('naive-l', False, ' & A & &'), ('naive-r', False, ' & & & A'), ('naive-l', True, '{} & {A} & {} & {}'), ('naive-r', True, '{} & {} & {} & {A}')])
def test_multicol_naive(df, multicol_align, siunitx, header):
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c')])
    df.columns = ridx
    level1 = ' & a & b & c' if not siunitx else '{} & {a} & {b} & {c}'
    col_format = 'lrrl' if not siunitx else 'lSSl'
    expected = dedent(f'        \\begin{{tabular}}{{{col_format}}}\n        {header} \\\\\n        {level1} \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\end{{tabular}}\n        ')
    styler = df.style.format(precision=2)
    result = styler.to_latex(multicol_align=multicol_align, siunitx=siunitx)
    assert expected == result