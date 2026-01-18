import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('level', [1, 'one', [1], ['one']])
def test_hide_index_level(mi_styler, level):
    mi_styler.index.names, mi_styler.columns.names = (['zero', 'one'], ['zero', 'one'])
    ctx = mi_styler.hide(axis='index', level=level)._translate(False, True)
    assert len(ctx['head'][0]) == 3
    assert len(ctx['head'][1]) == 3
    assert len(ctx['head'][2]) == 4
    assert ctx['head'][2][0]['is_visible']
    assert not ctx['head'][2][1]['is_visible']
    assert ctx['body'][0][0]['is_visible']
    assert not ctx['body'][0][1]['is_visible']
    assert ctx['body'][1][0]['is_visible']
    assert not ctx['body'][1][1]['is_visible']