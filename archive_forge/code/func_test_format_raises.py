import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('formatter', [5, True, [2.0]])
@pytest.mark.parametrize('func', ['format', 'format_index'])
def test_format_raises(styler, formatter, func):
    with pytest.raises(TypeError, match='expected str or callable'):
        getattr(styler, func)(formatter)