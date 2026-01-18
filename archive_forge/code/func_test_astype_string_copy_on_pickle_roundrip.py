import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_astype_string_copy_on_pickle_roundrip():
    base = Series(np.array([(1, 2), None, 1], dtype='object'))
    base_copy = pickle.loads(pickle.dumps(base))
    base_copy.astype(str)
    tm.assert_series_equal(base, base_copy)