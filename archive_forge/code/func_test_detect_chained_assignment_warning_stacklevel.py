from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('rhs', [3, DataFrame({0: [1, 2, 3, 4]})])
def test_detect_chained_assignment_warning_stacklevel(self, rhs, using_copy_on_write, warn_copy_on_write):
    df = DataFrame(np.arange(25).reshape(5, 5))
    df_original = df.copy()
    chained = df.loc[:3]
    with option_context('chained_assignment', 'warn'):
        if not using_copy_on_write and (not warn_copy_on_write):
            with tm.assert_produces_warning(SettingWithCopyWarning) as t:
                chained[2] = rhs
                assert t[0].filename == __file__
        else:
            chained[2] = rhs
            tm.assert_frame_equal(df, df_original)