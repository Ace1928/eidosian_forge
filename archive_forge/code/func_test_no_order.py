import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['polynomial', 'spline'])
def test_no_order(self, method):
    pytest.importorskip('scipy')
    s = Series([0, 1, np.nan, 3])
    msg = 'You must specify the order of the spline or polynomial'
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method=method)