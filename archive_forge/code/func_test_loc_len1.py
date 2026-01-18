import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_loc_len1(self, data):
    df = pd.DataFrame({'A': data})
    res = df.loc[[0], 'A']
    assert res.ndim == 1
    assert res._mgr.arrays[0].ndim == 1
    if hasattr(res._mgr, 'blocks'):
        assert res._mgr._block.ndim == 1