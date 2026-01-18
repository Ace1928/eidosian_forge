import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('msg,labels,level', [('labels \\[4\\] not found in level', 4, 'a'), ('labels \\[7\\] not found in level', 7, 'b')])
def test_drop_raise_exception_if_labels_not_in_level(msg, labels, level):
    mi = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=['a', 'b'])
    s = Series([10, 20, 30], index=mi)
    df = DataFrame([10, 20, 30], index=mi)
    with pytest.raises(KeyError, match=msg):
        s.drop(labels, level=level)
    with pytest.raises(KeyError, match=msg):
        df.drop(labels, level=level)