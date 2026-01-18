import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_highlight_null_subset(styler):
    result = styler.highlight_null(color='red', subset=['A']).highlight_null(color='green', subset=['B'])._compute().ctx
    expected = {(1, 0): [('background-color', 'red')], (1, 1): [('background-color', 'green')]}
    assert result == expected