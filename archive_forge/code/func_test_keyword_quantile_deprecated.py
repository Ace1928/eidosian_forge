import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_keyword_quantile_deprecated():
    ser = Series([1, 2, 3, 4])
    with tm.assert_produces_warning(FutureWarning):
        ser.expanding().quantile(quantile=0.5)