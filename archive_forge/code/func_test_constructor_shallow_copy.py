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
def test_constructor_shallow_copy(self):
    s = Series([1, 2, 3])
    s_orig = s.copy()
    s2 = Series(s)
    assert s2._mgr is not s._mgr
    s2.index = ['a', 'b', 'c']
    tm.assert_series_equal(s, s_orig)