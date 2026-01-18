import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_boolean_format():
    df = DataFrame([[True, False]])
    ctx = df.style._translate(True, True)
    assert ctx['body'][0][1]['display_value'] is True
    assert ctx['body'][0][2]['display_value'] is False