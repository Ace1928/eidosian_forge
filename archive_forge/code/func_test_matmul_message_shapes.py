import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_matmul_message_shapes(self):
    a = np.random.default_rng(2).random((10, 4))
    b = np.random.default_rng(2).random((5, 3))
    df = DataFrame(b)
    msg = 'shapes \\(10, 4\\) and \\(5, 3\\) not aligned'
    with pytest.raises(ValueError, match=msg):
        a @ df
    with pytest.raises(ValueError, match=msg):
        a.tolist() @ df