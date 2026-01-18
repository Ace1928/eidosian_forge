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
def test_hide_single_index(self, df):
    ctx = df.style._translate(True, True)
    assert ctx['body'][0][0]['is_visible']
    assert ctx['head'][0][0]['is_visible']
    ctx2 = df.style.hide(axis='index')._translate(True, True)
    assert not ctx2['body'][0][0]['is_visible']
    assert not ctx2['head'][0][0]['is_visible']
    ctx3 = df.set_index('A').style._translate(True, True)
    assert ctx3['body'][0][0]['is_visible']
    assert len(ctx3['head']) == 2
    assert ctx3['head'][0][0]['is_visible']
    ctx4 = df.set_index('A').style.hide(axis='index')._translate(True, True)
    assert not ctx4['body'][0][0]['is_visible']
    assert len(ctx4['head']) == 1
    assert not ctx4['head'][0][0]['is_visible']