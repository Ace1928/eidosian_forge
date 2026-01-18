from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
def test_constructor_range_values_mismatched_dtype(self, dtype):
    rng = Index(range(5))
    result = Index(rng, dtype=dtype)
    assert result.dtype == dtype
    result = Index(range(5), dtype=dtype)
    assert result.dtype == dtype