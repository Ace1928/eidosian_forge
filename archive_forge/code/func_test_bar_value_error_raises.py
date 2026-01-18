import io
import numpy as np
import pytest
from pandas import (
def test_bar_value_error_raises():
    df = DataFrame({'A': [-100, -60, -30, -20]})
    msg = "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(align='poorly', color=['#d65f5f', '#5fba7d']).to_html()
    msg = '`width` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(width=200).to_html()
    msg = '`height` must be a value in \\[0, 100\\]'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(height=200).to_html()