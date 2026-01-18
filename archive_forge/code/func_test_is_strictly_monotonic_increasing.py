import numpy as np
import pytest
from pandas import (
def test_is_strictly_monotonic_increasing():
    idx = MultiIndex(levels=[['bar', 'baz'], ['mom', 'next']], codes=[[0, 0, 1, 1], [0, 0, 0, 1]])
    assert idx.is_monotonic_increasing is True
    assert idx._is_strictly_monotonic_increasing is False