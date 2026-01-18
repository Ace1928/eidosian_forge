import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_mixing_cmap_and_colormap_raises(self):
    df = DataFrame({'A': np.random.default_rng(2).uniform(size=20), 'B': np.random.default_rng(2).uniform(size=20), 'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)})
    msg = 'Only specify one of `cmap` and `colormap`'
    with pytest.raises(TypeError, match=msg):
        df.plot.hexbin(x='A', y='B', cmap='YlGn', colormap='BuGn')