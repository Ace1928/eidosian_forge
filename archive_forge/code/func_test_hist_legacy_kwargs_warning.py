import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('kwargs', [{}, {'bins': 5}])
def test_hist_legacy_kwargs_warning(self, ts, kwargs):
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        _check_plot_works(ts.hist, by=ts.index.month, **kwargs)