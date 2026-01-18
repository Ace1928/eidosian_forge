import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_non_numeric_na():
    df = DataFrame({'object': [None, np.nan, 'foo'], 'datetime': [None, NaT, Timestamp('20120101')]})
    ctx = df.style.format(None, na_rep='-')._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '-'
    assert ctx['body'][0][2]['display_value'] == '-'
    assert ctx['body'][1][1]['display_value'] == '-'
    assert ctx['body'][1][2]['display_value'] == '-'