from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_categorical_to_object(self):
    ci = CategoricalIndex(range(5))
    result = Index(ci, dtype=object)
    assert not isinstance(result, CategoricalIndex)