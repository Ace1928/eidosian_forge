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
def test_freq_argument_required(self):
    msg = "missing 1 required positional argument: 'freq'"
    with pytest.raises(TypeError, match=msg):
        PeriodDtype()
    msg = 'PeriodDtype argument should be string or BaseOffset, got NoneType'
    with pytest.raises(TypeError, match=msg):
        PeriodDtype(None)