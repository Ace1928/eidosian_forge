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
@pytest.mark.parametrize('dtype', ['O', 'S', 'U'])
def test_array_equivalent_str(dtype):
    assert array_equivalent(np.array(['A', 'B'], dtype=dtype), np.array(['A', 'B'], dtype=dtype))
    assert not array_equivalent(np.array(['A', 'B'], dtype=dtype), np.array(['A', 'X'], dtype=dtype))