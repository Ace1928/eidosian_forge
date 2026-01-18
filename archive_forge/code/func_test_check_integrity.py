import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
def test_check_integrity(self):
    msg = 'Too many indices'
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=1, indices=[1, 2, 3])
    msg = 'No index can be less than zero'
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, -2, 3])
    msg = 'No index can be less than zero'
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, -2, 3])
    msg = 'All indices must be less than the length'
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, 2, 5])
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, 2, 6])
    msg = 'Indices must be strictly increasing'
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, 3, 2])
    with pytest.raises(ValueError, match=msg):
        IntIndex(length=5, indices=[1, 3, 3])