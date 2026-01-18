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
def test_hidden_column_names(mi_df):
    mi_df.columns.names = ['Lev0', 'Lev1']
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == 'Lev0'
    assert ctx['head'][1][1]['display_value'] == 'Lev1'
    mi_styler.hide(names=True, axis='columns')
    ctx = mi_styler._translate(True, True)
    assert ctx['head'][0][1]['display_value'] == '&nbsp;'
    assert ctx['head'][1][1]['display_value'] == '&nbsp;'
    mi_styler.hide(level=0, axis='columns')
    ctx = mi_styler._translate(True, True)
    assert len(ctx['head']) == 1
    assert ctx['head'][0][1]['display_value'] == '&nbsp;'