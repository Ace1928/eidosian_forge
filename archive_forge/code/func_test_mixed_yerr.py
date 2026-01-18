import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.xfail(reason='Open bug in matplotlib https://github.com/matplotlib/matplotlib/issues/11357')
def test_mixed_yerr(self):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    df = DataFrame([{'x': 1, 'a': 1, 'b': 1}, {'x': 2, 'a': 2, 'b': 3}])
    ax = df.plot('x', 'a', c='orange', yerr=0.1, label='orange')
    df.plot('x', 'b', c='blue', yerr=None, ax=ax, label='blue')
    legend = ax.get_legend()
    if Version(mpl.__version__) < Version('3.7'):
        result_handles = legend.legendHandles
    else:
        result_handles = legend.legend_handles
    assert isinstance(result_handles[0], LineCollection)
    assert isinstance(result_handles[1], Line2D)