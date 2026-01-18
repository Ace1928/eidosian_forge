from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_hide_columns_non_unique(styler):
    ctx = styler.hide(['d'], axis='columns')._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'c'
    assert ctx['head'][0][1]['is_visible'] is True
    assert ctx['head'][0][2]['display_value'] == 'd'
    assert ctx['head'][0][2]['is_visible'] is False
    assert ctx['head'][0][3]['display_value'] == 'd'
    assert ctx['head'][0][3]['is_visible'] is False
    assert ctx['body'][0][1]['is_visible'] is True
    assert ctx['body'][0][2]['is_visible'] is False
    assert ctx['body'][0][3]['is_visible'] is False