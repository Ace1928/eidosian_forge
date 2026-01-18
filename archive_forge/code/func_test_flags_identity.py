from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_flags_identity(self, frame_or_series):
    obj = Series([1, 2])
    if frame_or_series is DataFrame:
        obj = obj.to_frame()
    assert obj.flags is obj.flags
    obj2 = obj.copy()
    assert obj2.flags is not obj.flags