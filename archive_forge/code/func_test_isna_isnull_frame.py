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
@pytest.mark.parametrize('isna_f', [isna, isnull])
@pytest.mark.parametrize('data', [np.arange(4, dtype=float), [0.0, 1.0, 0.0, 1.0], Series(list('abcd'), dtype=object), date_range('2020-01-01', periods=4)])
@pytest.mark.parametrize('index', [date_range('2020-01-01', periods=4), range(4), period_range('2020-01-01', periods=4)])
def test_isna_isnull_frame(self, isna_f, data, index):
    df = pd.DataFrame(data, index=index)
    result = isna_f(df)
    expected = df.apply(isna_f)
    tm.assert_frame_equal(result, expected)