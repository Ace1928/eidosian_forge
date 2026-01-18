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
@pytest.mark.parametrize('stacked', [True, False])
def test_bar_linewidth(self, stacked):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot.bar(stacked=stacked, linewidth=2)
    for r in ax.patches:
        assert r.get_linewidth() == 2