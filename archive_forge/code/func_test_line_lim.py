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
@pytest.mark.parametrize('kwargs', [{}, {'secondary_y': True}])
def test_line_lim(self, kwargs):
    df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=['x', 'y', 'z'])
    ax = df.plot(**kwargs)
    xmin, xmax = ax.get_xlim()
    lines = ax.get_lines()
    assert xmin <= lines[0].get_data()[0][0]
    assert xmax >= lines[0].get_data()[0][-1]