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
def test_categorical_dtype_equality_requires_categories(self):
    first = CategoricalDtype(['a', 'b'])
    second = CategoricalDtype()
    third = CategoricalDtype(ordered=True)
    assert second == second
    assert third == third
    assert first != second
    assert second != first
    assert first != third
    assert third != first
    assert second == third
    assert third == second