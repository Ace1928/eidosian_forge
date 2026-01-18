from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.fixture
def styler_mi():
    midx = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    return Styler(DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx))