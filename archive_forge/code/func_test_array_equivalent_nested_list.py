from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('strict_nan', [True, False])
def test_array_equivalent_nested_list(strict_nan):
    left = np.array([[50, 70, 90], [20, 30]], dtype=object)
    right = np.array([[50, 70, 90], [20, 30]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)
    left = np.array([[50, 50, 50], [40, 40]], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)