import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_long_int_formatting():
    df = DataFrame(data=[[1234567890123456789]], columns=['test'])
    styler = df.style
    ctx = styler._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '1234567890123456789'
    styler = df.style.format(thousands='_')
    ctx = styler._translate(True, True)
    assert ctx['body'][0][1]['display_value'] == '1_234_567_890_123_456_789'