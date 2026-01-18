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
def test_from_categorical_dtype_both(self):
    c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
    result = CategoricalDtype._from_categorical_dtype(c1, categories=[1, 2], ordered=False)
    assert result == CategoricalDtype([1, 2], ordered=False)