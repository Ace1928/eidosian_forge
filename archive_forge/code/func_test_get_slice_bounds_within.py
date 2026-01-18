import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
def test_get_slice_bounds_within(self, side, expected):
    index = Index(list('abcdef'))
    result = index.get_slice_bound('e', side=side)
    assert result == expected