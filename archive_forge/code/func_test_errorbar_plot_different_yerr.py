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
@pytest.mark.slow
@pytest.mark.parametrize('kind', ['line', 'bar', 'barh'])
@pytest.mark.parametrize('y_err', [Series(np.ones(12) * 0.2, name='x'), DataFrame({'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4})])
def test_errorbar_plot_different_yerr(self, kind, y_err):
    df = DataFrame({'x': np.arange(12), 'y': np.arange(12, 0, -1)})
    ax = _check_plot_works(df.plot, yerr=y_err, kind=kind)
    _check_has_errorbars(ax, xerr=0, yerr=2)