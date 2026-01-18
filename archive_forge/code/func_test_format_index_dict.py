import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_format_index_dict(styler):
    ctx = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(['X', 'Y']):
        assert ctx['body'][i][0]['display_value'] == val