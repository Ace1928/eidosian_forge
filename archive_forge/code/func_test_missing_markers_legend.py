import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_missing_markers_legend(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), columns=['A', 'B', 'C'])
    ax = df.plot(y=['A'], marker='x', linestyle='solid')
    df.plot(y=['B'], marker='o', linestyle='dotted', ax=ax)
    df.plot(y=['C'], marker='<', linestyle='dotted', ax=ax)
    _check_legend_labels(ax, labels=['A', 'B', 'C'])
    _check_legend_marker(ax, expected_markers=['x', 'o', '<'])