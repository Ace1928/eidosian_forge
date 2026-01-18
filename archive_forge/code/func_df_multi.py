import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.fixture
def df_multi():
    return DataFrame(data=np.arange(16).reshape(4, 4), columns=MultiIndex.from_product([['A', 'B'], ['a', 'b']]), index=MultiIndex.from_product([['X', 'Y'], ['x', 'y']]))