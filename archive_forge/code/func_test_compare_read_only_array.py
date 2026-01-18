from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_read_only_array():
    arr = np.array([], dtype=object)
    arr.flags.writeable = False
    idx = pd.Index(arr)
    result = idx > 69
    assert result.dtype == bool