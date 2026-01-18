import os
import pytest
import numpy as np
from . import util
@pytest.mark.slow
def test_constant_real_double(self):
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, self.module.foo_double, x)
    x = np.arange(3, dtype=np.float64)
    self.module.foo_double(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])