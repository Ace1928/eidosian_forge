import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('color, expected', [('green', ['green'] * 4), (['yellow', 'red', 'green', 'blue'], ['yellow', 'red', 'green', 'blue'])])
def test_color_and_marker(self, color, expected):
    df = DataFrame(np.random.default_rng(2).random((7, 4)))
    ax = df.plot(color=color, style='d--')
    result = [i.get_color() for i in ax.lines]
    assert result == expected
    assert all((i.get_linestyle() == '--' for i in ax.lines))
    assert all((i.get_marker() == 'd' for i in ax.lines))