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
@pytest.mark.parametrize('meth, dim', [('bar', 'get_width'), ('barh', 'get_height')])
@pytest.mark.parametrize('stacked', [True, False])
def test_bar_barwidth(self, meth, dim, stacked):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    width = 0.9
    ax = getattr(df.plot, meth)(stacked=stacked, width=width)
    for r in ax.patches:
        if not stacked:
            assert getattr(r, dim)() == width / len(df.columns)
        else:
            assert getattr(r, dim)() == width