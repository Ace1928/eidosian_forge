import re
import sys
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_duplicated_implemented_no_recursion():
    df = DataFrame(np.random.default_rng(2).integers(0, 1000, (10, 1000)))
    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        result = df.duplicated()
    finally:
        sys.setrecursionlimit(rec_limit)
    assert isinstance(result, Series)
    assert result.dtype == np.bool_