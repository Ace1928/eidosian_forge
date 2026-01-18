from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('ddof', range(3))
def test_nanstd_roundoff(self, ddof):
    data = Series(766897346 * np.ones(10))
    result = data.std(ddof=ddof)
    assert result == 0.0