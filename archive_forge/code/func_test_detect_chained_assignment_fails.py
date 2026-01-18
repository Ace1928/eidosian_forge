from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_fails(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': Series(range(2), dtype='int64'), 'B': np.array(np.arange(2, 4), dtype=np.float64)})
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df.loc[0]['A'] = -5
    else:
        with pytest.raises(SettingWithCopyError, match=msg):
            df.loc[0]['A'] = -5