import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('kwargs', [{}, {'grid': False}, {'figsize': (8, 10)}])
def test_hist_legacy_kwargs(self, ts, kwargs):
    _check_plot_works(ts.hist, **kwargs)