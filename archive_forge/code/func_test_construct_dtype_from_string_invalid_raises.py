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
@pytest.mark.parametrize('string', ['foo', 'period[foo]', 'foo[D]', 'datetime64[ns]', 'datetime64[ns, US/Eastern]'])
def test_construct_dtype_from_string_invalid_raises(self, string):
    msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
    with pytest.raises(TypeError, match=re.escape(msg)):
        PeriodDtype.construct_from_string(string)