import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_complex_step(self):
    g = r_[0:36:100j]
    assert_(g.shape == (100,))
    g = r_[0:36:np.complex64(100j)]
    assert_(g.shape == (100,))