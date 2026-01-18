import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_highlight_between_raises2(styler):
    msg = "values can be 'both', 'left', 'right', or 'neither'"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive='badstring')._compute()
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive=1)._compute()