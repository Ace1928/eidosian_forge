import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_dataframe_keys_bug(self, sort):
    t1 = DataFrame({'value': Series([1, 2, 3], index=Index(['a', 'b', 'c'], name='id'))})
    t2 = DataFrame({'value': Series([7, 8], index=Index(['a', 'b'], name='id'))})
    result = concat([t1, t2], axis=1, keys=['t1', 't2'], sort=sort)
    assert list(result.columns) == [('t1', 'value'), ('t2', 'value')]