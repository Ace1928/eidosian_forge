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
def test_line_area_stacked_mixed(self):
    mixed_df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['w', 'x', 'y', 'z'])
    _check_plot_works(mixed_df.plot, stacked=False)
    msg = "When stacked is True, each column must be either all positive or all negative. Column 'w' contains both positive and negative values"
    with pytest.raises(ValueError, match=msg):
        mixed_df.plot(stacked=True)