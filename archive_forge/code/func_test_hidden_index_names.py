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
def test_hidden_index_names(mi_df):
    mi_df.index.names = ['Lev0', 'Lev1']
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 3
    mi_styler.hide(axis='index', names=True)
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 2
    for i in range(4):
        assert ctx['body'][0][i]['is_visible']
    mi_styler.hide(axis='index', level=1)
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 2
    assert ctx['body'][0][0]['is_visible'] is True
    assert ctx['body'][0][1]['is_visible'] is False