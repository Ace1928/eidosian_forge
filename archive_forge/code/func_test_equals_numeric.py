import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_numeric(self):
    index_cls = Index
    idx = index_cls([1.0, 2.0])
    assert idx.equals(idx)
    assert idx.identical(idx)
    idx2 = index_cls([1.0, 2.0])
    assert idx.equals(idx2)
    idx = index_cls([1.0, np.nan])
    assert idx.equals(idx)
    assert idx.identical(idx)
    idx2 = index_cls([1.0, np.nan])
    assert idx.equals(idx2)