import os
import pytest
import numpy as np
from . import util
@pytest.mark.slow
def test_constant_no(self):
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, self.module.foo_no, x)
    x = np.arange(3, dtype=np.float64)
    self.module.foo_no(x)
    assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])