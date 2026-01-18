from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_implicit_take2(self, using_copy_on_write, warn_copy_on_write):
    if using_copy_on_write or warn_copy_on_write:
        pytest.skip('_is_copy is not always set for CoW')
    df = random_text(100000)
    indexer = df.letters.apply(lambda x: len(x) > 10)
    df = df.loc[indexer]
    assert df._is_copy is not None
    df.loc[:, 'letters'] = df['letters'].apply(str.lower)
    assert df._is_copy is not None
    df['letters'] = df['letters'].apply(str.lower)
    assert df._is_copy is None