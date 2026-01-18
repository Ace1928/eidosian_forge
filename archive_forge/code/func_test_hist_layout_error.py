import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_layout_error(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    msg = 'Layout of 1x1 must be larger than required size 3'
    with pytest.raises(ValueError, match=msg):
        df.hist(layout=(1, 1))
    msg = re.escape('Layout must be a tuple of (rows, columns)')
    with pytest.raises(ValueError, match=msg):
        df.hist(layout=(1,))
    msg = 'At least one dimension of layout must be positive'
    with pytest.raises(ValueError, match=msg):
        df.hist(layout=(-1, -1))