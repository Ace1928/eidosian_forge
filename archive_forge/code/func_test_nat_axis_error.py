import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('msg, axis', [['column', 1], ['index', 0]])
def test_nat_axis_error(msg, axis):
    idx = [Timestamp('2020'), NaT]
    kwargs = {'columns' if axis == 1 else 'index': idx}
    df = DataFrame(np.eye(2), **kwargs)
    warn_msg = "The 'axis' keyword in DataFrame.rolling is deprecated"
    if axis == 1:
        warn_msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with pytest.raises(ValueError, match=f'{msg} values must not have NaT'):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            df.rolling('D', axis=axis).mean()