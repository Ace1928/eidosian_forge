import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_uints():
    return DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=Index(range(0, 8, 2), dtype=np.uint64), columns=Index(range(0, 12, 3), dtype=np.uint64))