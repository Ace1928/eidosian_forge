import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@td.skip_array_manager_invalid_test
def test_astype_numpy_to_ea():
    ser = Series([1, 2, 3])
    with pd.option_context('mode.copy_on_write', True):
        result = ser.astype('Int64')
    assert np.shares_memory(get_array(ser), get_array(result))