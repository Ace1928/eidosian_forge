import numpy as np
import pytest
from pandas.core.indexers import (
def test_length_of_indexer():
    arr = np.zeros(4, dtype=bool)
    arr[0] = 1
    result = length_of_indexer(arr)
    assert result == 1