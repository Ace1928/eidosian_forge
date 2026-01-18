import os
import pytest
import numpy as np
from . import util
@pytest.mark.slow
def test_constant_non_compound_int(self):
    x = np.arange(4, dtype=np.int32)
    self.module.foo_non_compound_int(x)
    assert np.allclose(x, [0 + 1 + 2 + 3 * 4, 1, 2, 3])