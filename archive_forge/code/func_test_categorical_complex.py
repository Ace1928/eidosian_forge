import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_categorical_complex():
    result = Categorical([1, 2 + 2j])
    expected = Categorical([1.0 + 0j, 2.0 + 2j])
    tm.assert_categorical_equal(result, expected)
    result = Categorical([1, 2, 2 + 2j])
    expected = Categorical([1.0 + 0j, 2.0 + 0j, 2.0 + 2j])
    tm.assert_categorical_equal(result, expected)