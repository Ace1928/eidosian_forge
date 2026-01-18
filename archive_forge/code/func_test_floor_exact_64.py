import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_floor_exact_64():
    for e in range(53, 63):
        start = np.float64(2 ** e)
        across = start + np.arange(2048, dtype=np.float64)
        gaps = set(np.diff(across)).difference([0])
        assert len(gaps) == 1
        gap = gaps.pop()
        assert gap == int(gap)
        test_val = 2 ** (e + 1) - 1
        assert floor_exact(test_val, np.float64) == 2 ** (e + 1) - int(gap)