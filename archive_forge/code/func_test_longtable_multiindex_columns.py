from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('sparse, exp, siunitx', [(True, '{} & \\multicolumn{2}{r}{A} & {B}', True), (False, '{} & {A} & {A} & {B}', True), (True, ' & \\multicolumn{2}{r}{A} & B', False), (False, ' & A & A & B', False)])
def test_longtable_multiindex_columns(df, sparse, exp, siunitx):
    cidx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df.columns = cidx
    with_si = '{} & {a} & {b} & {c} \\\\'
    without_si = ' & a & b & c \\\\'
    expected = dedent(f'        \\begin{{longtable}}{{l{('SS' if siunitx else 'rr')}l}}\n        {exp} \\\\\n        {(with_si if siunitx else without_si)}\n        \\endfirsthead\n        {exp} \\\\\n        {(with_si if siunitx else without_si)}\n        \\endhead\n        ')
    result = df.style.to_latex(environment='longtable', sparse_columns=sparse, siunitx=siunitx)
    assert expected in result