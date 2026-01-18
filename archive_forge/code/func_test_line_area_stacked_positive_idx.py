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
@pytest.mark.parametrize('kind', ['line', 'area'])
def test_line_area_stacked_positive_idx(self, kind):
    df = DataFrame(np.random.default_rng(2).random((6, 4)), columns=['w', 'x', 'y', 'z'])
    df2 = df.set_index(df.index + 1)
    _check_plot_works(df2.plot, kind=kind, logx=True, stacked=True)