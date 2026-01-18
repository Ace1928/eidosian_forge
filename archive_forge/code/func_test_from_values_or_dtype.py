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
@pytest.mark.parametrize('values, categories, ordered, dtype, expected', [[None, None, None, None, CategoricalDtype()], [None, ['a', 'b'], True, None, dtype1], [c, None, None, dtype2, dtype2], [c, ['x', 'y'], False, None, dtype2]])
def test_from_values_or_dtype(self, values, categories, ordered, dtype, expected):
    result = CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
    assert result == expected