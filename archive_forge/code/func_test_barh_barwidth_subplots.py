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
def test_barh_barwidth_subplots(self, meth, dim):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    width = 0.9
    axes = getattr(df.plot, meth)(width=width, subplots=True)
    for ax in axes:
        for r in ax.patches:
            assert getattr(r, dim)() == width