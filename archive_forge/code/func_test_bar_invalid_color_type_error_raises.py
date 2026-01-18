import io
import numpy as np
import pytest
from pandas import (
def test_bar_invalid_color_type_error_raises():
    df = DataFrame({'A': [1, 2, 3, 4]})
    msg = "`color` must be string or list or tuple of 2 strings,\\(eg: color=\\['#d65f5f', '#5fba7d'\\]\\)"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=123).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=['#d65f5f', '#5fba7d', '#abcdef']).to_html()