from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_uses_size_on_exception():

    class RaisingObjectException(Exception):
        pass

    class RaisingObject:

        def __init__(self, msg='I will raise inside Cython') -> None:
            super().__init__()
            self.msg = msg

        def __eq__(self, other):
            raise RaisingObjectException(self.msg)
    df = DataFrame({'a': [RaisingObject() for _ in range(4)], 'grp': list('ab' * 2)})
    result = df.groupby('grp').count()
    expected = DataFrame({'a': [2, 2]}, index=Index(list('ab'), name='grp'))
    tm.assert_frame_equal(result, expected)