import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
@pytest.mark.parametrize('na_action', [None, 'ignore'])
def test_map_raises(self, data, na_action):
    msg = 'fill value in the sparse values not supported'
    with pytest.raises(ValueError, match=msg):
        data.map(lambda x: np.nan, na_action=na_action)