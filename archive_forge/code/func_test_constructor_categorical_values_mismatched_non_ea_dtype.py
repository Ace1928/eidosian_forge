from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [object, 'float64', 'uint64', 'category'])
def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype):
    cat = Categorical([1, 2, 3])
    result = Index(cat, dtype=dtype)
    assert result.dtype == dtype