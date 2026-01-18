from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_setitem_chained_assignment(self, using_copy_on_write):
    with option_context('chained_assignment', None):
        df = DataFrame({'aa': range(5), 'bb': [2.2] * 5})
        df['cc'] = 0.0
        ck = [True] * len(df)
        with tm.raises_chained_assignment_error():
            df['bb'].iloc[0] = 0.13
        df.iloc[ck]
        with tm.raises_chained_assignment_error():
            df['bb'].iloc[0] = 0.15
        if not using_copy_on_write:
            assert df['bb'].iloc[0] == 0.15
        else:
            assert df['bb'].iloc[0] == 2.2