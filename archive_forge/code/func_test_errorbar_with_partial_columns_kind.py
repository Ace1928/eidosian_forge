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
@pytest.mark.parametrize('kind', ['line', 'bar'])
def test_errorbar_with_partial_columns_kind(self, kind):
    df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
    df_err = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2])
    ax = _check_plot_works(df.plot, yerr=df_err, kind=kind)
    _check_has_errorbars(ax, xerr=0, yerr=2)