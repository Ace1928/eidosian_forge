import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_bins_legacy(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    ax = df.hist(bins=2)[0][0]
    assert len(ax.patches) == 2