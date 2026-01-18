from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('val', [3075.2, 3075.3, 3075.5])
def test_constant_series(self, val):
    data = val * np.ones(300)
    kurt = nanops.nankurt(data)
    assert kurt == 0.0