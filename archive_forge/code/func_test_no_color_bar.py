import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_no_color_bar(self):
    df = DataFrame({'A': np.random.default_rng(2).uniform(size=20), 'B': np.random.default_rng(2).uniform(size=20), 'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)})
    ax = df.plot.hexbin(x='A', y='B', colorbar=None)
    assert ax.collections[0].colorbar is None