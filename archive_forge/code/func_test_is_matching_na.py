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
def test_is_matching_na(self, nulls_fixture, nulls_fixture2):
    left = nulls_fixture
    right = nulls_fixture2
    assert libmissing.is_matching_na(left, left)
    if left is right:
        assert libmissing.is_matching_na(left, right)
    elif is_float(left) and is_float(right):
        assert libmissing.is_matching_na(left, right)
    elif type(left) is type(right):
        assert libmissing.is_matching_na(left, right)
    else:
        assert not libmissing.is_matching_na(left, right)