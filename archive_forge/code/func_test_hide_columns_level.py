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
@pytest.mark.parametrize('names', [True, False])
def test_hide_columns_level(mi_styler, level, names):
    mi_styler.columns.names = ['zero', 'one']
    if names:
        mi_styler.index.names = ['zero', 'one']
    ctx = mi_styler.hide(axis='columns', level=level)._translate(True, False)
    assert len(ctx['head']) == (2 if names else 1)