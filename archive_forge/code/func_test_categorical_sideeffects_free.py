from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_categorical_sideeffects_free(self):
    cat = Categorical(['a', 'b', 'c', 'a'])
    s = Series(cat, copy=True)
    assert s.cat is not cat
    s = s.cat.rename_categories([1, 2, 3])
    exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
    exp_cat = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
    tm.assert_numpy_array_equal(s.__array__(), exp_s)
    tm.assert_numpy_array_equal(cat.__array__(), exp_cat)
    s[0] = 2
    exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
    tm.assert_numpy_array_equal(s.__array__(), exp_s2)
    tm.assert_numpy_array_equal(cat.__array__(), exp_cat)
    cat = Categorical(['a', 'b', 'c', 'a'])
    s = Series(cat, copy=False)
    assert s.values is cat
    s = s.cat.rename_categories([1, 2, 3])
    assert s.values is not cat
    exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
    tm.assert_numpy_array_equal(s.__array__(), exp_s)
    s[0] = 2
    exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
    tm.assert_numpy_array_equal(s.__array__(), exp_s2)