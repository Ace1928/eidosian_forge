import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('consolidate', [True, False])
def test_loc_setitem_multiindex_columns(self, consolidate):
    A = DataFrame(np.zeros((6, 5), dtype=np.float32))
    A = pd.concat([A, A], axis=1, keys=[1, 2])
    if consolidate:
        A = A._consolidate()
    A.loc[2:3, (1, slice(2, 3))] = np.ones((2, 2), dtype=np.float32)
    assert (A.dtypes == np.float32).all()
    A.loc[0:5, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)
    assert (A.dtypes == np.float32).all()
    A.loc[:, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)
    assert (A.dtypes == np.float32).all()