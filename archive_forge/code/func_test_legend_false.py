import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_legend_false(self):
    df = DataFrame({'a': [1, 1], 'b': [2, 3]})
    df2 = DataFrame({'d': [2.5, 2.5]})
    ax = df.plot(legend=True, color={'a': 'blue', 'b': 'green'}, secondary_y='b')
    df2.plot(legend=True, color={'d': 'red'}, ax=ax)
    legend = ax.get_legend()
    if Version(mpl.__version__) < Version('3.7'):
        handles = legend.legendHandles
    else:
        handles = legend.legend_handles
    result = [handle.get_color() for handle in handles]
    expected = ['blue', 'green', 'red']
    assert result == expected