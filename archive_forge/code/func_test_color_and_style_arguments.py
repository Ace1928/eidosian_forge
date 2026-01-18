import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_color_and_style_arguments(self):
    df = DataFrame({'x': [1, 2], 'y': [3, 4]})
    ax = df.plot(color=['red', 'black'], style=['-', '--'])
    linestyle = [line.get_linestyle() for line in ax.lines]
    assert linestyle == ['-', '--']
    color = [line.get_color() for line in ax.lines]
    assert color == ['red', 'black']
    msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
    with pytest.raises(ValueError, match=msg):
        df.plot(color=['red', 'black'], style=['k-', 'r--'])