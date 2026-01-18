import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_is_strictly_monotonic(self):
    index_cls = Index
    index = index_cls([1, 1, 2, 3])
    assert index.is_monotonic_increasing is True
    assert index._is_strictly_monotonic_increasing is False
    index = index_cls([3, 2, 1, 1])
    assert index.is_monotonic_decreasing is True
    assert index._is_strictly_monotonic_decreasing is False
    index = index_cls([1, 1])
    assert index.is_monotonic_increasing
    assert index.is_monotonic_decreasing
    assert not index._is_strictly_monotonic_increasing
    assert not index._is_strictly_monotonic_decreasing