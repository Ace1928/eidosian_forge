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
@pytest.mark.parametrize('mult', [1, -1])
def test_line_area_stacked(self, kind, mult):
    df = mult * DataFrame(np.random.default_rng(2).random((6, 4)), columns=['w', 'x', 'y', 'z'])
    ax1 = _check_plot_works(df.plot, kind=kind, stacked=False)
    ax2 = _check_plot_works(df.plot, kind=kind, stacked=True)
    self._compare_stacked_y_cood(ax1.lines, ax2.lines)