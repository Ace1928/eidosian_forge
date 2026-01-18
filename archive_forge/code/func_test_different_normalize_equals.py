from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def test_different_normalize_equals(self, _offset):
    offset = _offset()
    offset2 = _offset(normalize=True)
    assert offset != offset2