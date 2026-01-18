import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('w', [1, 1.0])
def test_bar_barwidth_position_int(self, w):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot.bar(stacked=True, width=w)
    ticks = ax.xaxis.get_ticklocs()
    tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))
    assert ax.get_xlim() == (-0.75, 4.75)
    assert ax.patches[0].get_x() == -0.5
    assert ax.patches[-1].get_x() == 3.5