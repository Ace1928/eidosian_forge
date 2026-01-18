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
@pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
def test_name_repr_generic(self, subtype):
    dtype = IntervalDtype(subtype)
    assert str(dtype) == 'interval'
    assert dtype.name == 'interval'