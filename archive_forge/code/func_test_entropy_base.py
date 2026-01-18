import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_entropy_base(self):
    pk = np.ones(16, float)
    S = stats.entropy(pk, base=2.0)
    assert_(abs(S - 4.0) < 1e-05)
    qk = np.ones(16, float)
    qk[:8] = 2.0
    S = stats.entropy(pk, qk)
    S2 = stats.entropy(pk, qk, base=2.0)
    assert_(abs(S / S2 - np.log(2.0)) < 1e-05)