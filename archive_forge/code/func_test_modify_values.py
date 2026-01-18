from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_modify_values(self, float_frame, using_copy_on_write):
    if using_copy_on_write:
        with pytest.raises(ValueError, match='read-only'):
            float_frame.values[5] = 5
        assert (float_frame.values[5] != 5).all()
        return
    float_frame.values[5] = 5
    assert (float_frame.values[5] == 5).all()
    float_frame['E'] = 7.0
    col = float_frame['E']
    float_frame.values[6] = 6
    assert not (float_frame.values[6] == 6).all()
    assert (col == 7).all()