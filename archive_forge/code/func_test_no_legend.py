import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('kind', ['line', 'bar', 'barh', pytest.param('kde', marks=td.skip_if_no('scipy')), 'area', 'hist'])
def test_no_legend(self, kind):
    df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=['a', 'b', 'c'])
    ax = df.plot(kind=kind, legend=False)
    _check_legend_labels(ax, visible=False)