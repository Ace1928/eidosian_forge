from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_isin_nan_common_object(self, nulls_fixture, nulls_fixture2, using_infer_string):
    idx = Index(['a', nulls_fixture])
    if isinstance(nulls_fixture, float) and isinstance(nulls_fixture2, float) and math.isnan(nulls_fixture) and math.isnan(nulls_fixture2):
        tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
    elif nulls_fixture is nulls_fixture2:
        tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
    elif using_infer_string and idx.dtype == 'string':
        tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
    else:
        tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, False]))