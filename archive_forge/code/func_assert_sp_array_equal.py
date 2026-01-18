from __future__ import annotations
import operator
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import (
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing
def assert_sp_array_equal(left, right) -> None:
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    _check_isinstance(left, right, pd.arrays.SparseArray)
    assert_numpy_array_equal(left.sp_values, right.sp_values)
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)
    left_index = left.sp_index
    right_index = right.sp_index
    if not left_index.equals(right_index):
        raise_assert_detail('SparseArray.index', 'index are not equal', left_index, right_index)
    else:
        pass
    assert_attr_equal('fill_value', left, right)
    assert_attr_equal('dtype', left, right)
    assert_numpy_array_equal(left.to_dense(), right.to_dense())