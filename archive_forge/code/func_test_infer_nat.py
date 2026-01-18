from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [NaT, None, np.nan, float('nan')])
def test_infer_nat(self, val):
    values = [NaT, val]
    idx = Index(values)
    assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
    idx = Index(values[::-1])
    assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
    idx = Index(np.array(values, dtype=object))
    assert idx.dtype == 'datetime64[ns]' and idx.isna().all()
    idx = Index(np.array(values, dtype=object)[::-1])
    assert idx.dtype == 'datetime64[ns]' and idx.isna().all()