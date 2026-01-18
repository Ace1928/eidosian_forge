from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_hexbin_basic_subplots(self):
    df = DataFrame({'A': np.random.default_rng(2).uniform(size=20), 'B': np.random.default_rng(2).uniform(size=20), 'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)})
    axes = df.plot.hexbin(x='A', y='B', subplots=True)
    assert len(axes[0].figure.axes) == 2
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))