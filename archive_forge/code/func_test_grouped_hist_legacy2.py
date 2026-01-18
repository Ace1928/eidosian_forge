import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_grouped_hist_legacy2(self):
    n = 10
    weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
    height = Series(np.random.default_rng(2).normal(60, 10, size=n))
    gender_int = np.random.default_rng(2).choice([0, 1], size=n)
    df_int = DataFrame({'height': height, 'weight': weight, 'gender': gender_int})
    gb = df_int.groupby('gender')
    axes = gb.hist()
    assert len(axes) == 2
    assert len(mpl.pyplot.get_fignums()) == 2