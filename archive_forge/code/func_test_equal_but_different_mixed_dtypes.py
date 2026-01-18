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
def test_equal_but_different_mixed_dtypes(self):
    c1 = CategoricalDtype([1, 2, '3'])
    c2 = CategoricalDtype(['3', 1, 2])
    assert c1 is not c2
    assert c1 == c2