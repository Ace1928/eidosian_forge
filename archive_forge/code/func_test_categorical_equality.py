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
@pytest.mark.parametrize('ordered1', [True, False, None])
@pytest.mark.parametrize('ordered2', [True, False, None])
def test_categorical_equality(self, ordered1, ordered2):
    c1 = CategoricalDtype(list('abc'), ordered1)
    c2 = CategoricalDtype(list('abc'), ordered2)
    result = c1 == c2
    expected = bool(ordered1) is bool(ordered2)
    assert result is expected
    c1 = CategoricalDtype(list('abc'), ordered1)
    c2 = CategoricalDtype(list('cab'), ordered2)
    result = c1 == c2
    expected = bool(ordered1) is False and bool(ordered2) is False
    assert result is expected
    c2 = CategoricalDtype([1, 2, 3], ordered2)
    assert c1 != c2
    c1 = CategoricalDtype(list('abc'), ordered1)
    c2 = CategoricalDtype(None, ordered2)
    c3 = CategoricalDtype(None, ordered1)
    assert c1 != c2
    assert c2 != c1
    assert c2 == c3