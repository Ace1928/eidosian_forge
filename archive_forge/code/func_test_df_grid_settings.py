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
def test_df_grid_settings(self):
    _check_grid_settings(DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]}), plotting.PlotAccessor._dataframe_kinds, kws={'x': 'a', 'y': 'b'})