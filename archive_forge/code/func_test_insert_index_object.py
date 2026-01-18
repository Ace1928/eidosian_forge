from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, object), (1.1, 1.1, object), (False, False, object), ('x', 'x', object)])
def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
    obj = pd.Index(list('abcd'), dtype=object)
    assert obj.dtype == object
    exp = pd.Index(['a', coerced_val, 'b', 'c', 'd'], dtype=object)
    self._assert_insert_conversion(obj, insert, exp, coerced_dtype)