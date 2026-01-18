import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_options():
    df = DataFrame({'int': [2000, 1], 'float': [1.009, None], 'str': ['&<', '&~']})
    ctx = df.style._translate(True, True)
    assert ctx['body'][1][2]['display_value'] == 'nan'
    with option_context('styler.format.na_rep', 'MISSING'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][2]['display_value'] == 'MISSING'
    assert ctx['body'][0][2]['display_value'] == '1.009000'
    with option_context('styler.format.decimal', '_'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][2]['display_value'] == '1_009000'
    with option_context('styler.format.precision', 2):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][2]['display_value'] == '1.01'
    assert ctx['body'][0][1]['display_value'] == '2000'
    with option_context('styler.format.thousands', '_'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][1]['display_value'] == '2_000'
    assert ctx['body'][0][3]['display_value'] == '&<'
    assert ctx['body'][1][3]['display_value'] == '&~'
    with option_context('styler.format.escape', 'html'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][3]['display_value'] == '&amp;&lt;'
    with option_context('styler.format.escape', 'latex'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][3]['display_value'] == '\\&\\textasciitilde '
    with option_context('styler.format.escape', 'latex-math'):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][1][3]['display_value'] == '\\&\\textasciitilde '
    with option_context('styler.format.formatter', {'int': '{:,.2f}'}):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op['body'][0][1]['display_value'] == '2,000.00'