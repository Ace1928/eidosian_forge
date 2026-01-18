import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('arg, map, axis', [('left', [1, 2], 0), ('left', [1, 2, 3], 1), ('left', np.array([[1, 2], [1, 2]]), None), ('right', [1, 2], 0), ('right', [1, 2, 3], 1), ('right', np.array([[1, 2], [1, 2]]), None)])
def test_highlight_between_raises(arg, styler, map, axis):
    msg = f"supplied '{arg}' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(**{arg: map, 'axis': axis})._compute()