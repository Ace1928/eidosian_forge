import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('data, bound, expected', [(list('abcdef'), 'x', 6), (list('bcdefg'), 'a', 0)])
def test_get_slice_bounds_outside(self, side, expected, data, bound):
    index = Index(data)
    result = index.get_slice_bound(bound, side=side)
    assert result == expected