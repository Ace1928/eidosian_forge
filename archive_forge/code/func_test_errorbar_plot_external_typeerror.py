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
def test_errorbar_plot_external_typeerror(self):
    d = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
    df = DataFrame(d)
    df_err = DataFrame({'x': ['zzz'] * 12, 'y': ['zzz'] * 12})
    with tm.external_error_raised(TypeError):
        df.plot(yerr=df_err)