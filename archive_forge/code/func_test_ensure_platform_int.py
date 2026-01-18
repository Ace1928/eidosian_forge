from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_ensure_platform_int():
    arr = np.arange(100, dtype=np.intp)
    result = libalgos.ensure_platform_int(arr)
    assert result is arr