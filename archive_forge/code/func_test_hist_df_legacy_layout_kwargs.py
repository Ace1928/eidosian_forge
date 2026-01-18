import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('kwargs', [{'sharex': True, 'sharey': True}, {'figsize': (8, 10)}, {'bins': 5}])
def test_hist_df_legacy_layout_kwargs(self, kwargs):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    df[5] = to_datetime(np.random.default_rng(2).integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        _check_plot_works(df.hist, **kwargs)