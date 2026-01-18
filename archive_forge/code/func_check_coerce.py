import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def check_coerce(self, a, b, is_float_index=True):
    assert a.equals(b)
    tm.assert_index_equal(a, b, exact=False)
    if is_float_index:
        assert isinstance(b, Index)
    else:
        assert type(b) is Index