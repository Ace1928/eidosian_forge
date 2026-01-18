import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
@pytest.mark.parametrize('index', [Index(np.arange(5), dtype=np.int64), RangeIndex(5)])
def test_float_slice_getitem_with_integer_index_raises(self, idx, index):
    s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)
    sc = s.copy()
    sc.loc[idx] = 0
    result = sc.loc[idx].values.ravel()
    assert (result == 0).all()
    msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[(3|4)\\.0\\] of type float'
    with pytest.raises(TypeError, match=msg):
        s[idx] = 0
    with pytest.raises(TypeError, match=msg):
        s[idx]