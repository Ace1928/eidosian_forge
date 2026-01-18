import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
def test_array_to_timedelta64_non_object_raises(self):
    values = np.arange(5)
    msg = "'values' must have object dtype"
    with pytest.raises(TypeError, match=msg):
        array_to_timedelta64(values)