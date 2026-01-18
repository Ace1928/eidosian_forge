import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('n', [10, np.array([10, 10]), np.array([[[10]], [[10]]])])
def test_multinomial_pval_broadcast(self, n):
    random = Generator(MT19937(self.seed))
    pvals = np.array([1 / 4] * 4)
    actual = random.multinomial(n, pvals)
    n_shape = tuple() if isinstance(n, int) else n.shape
    expected_shape = n_shape + (4,)
    assert actual.shape == expected_shape
    pvals = np.vstack([pvals, pvals])
    actual = random.multinomial(n, pvals)
    expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1]) + (4,)
    assert actual.shape == expected_shape
    pvals = np.vstack([[pvals], [pvals]])
    actual = random.multinomial(n, pvals)
    expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1])
    assert actual.shape == expected_shape + (4,)
    actual = random.multinomial(n, pvals, size=(3, 2) + expected_shape)
    assert actual.shape == (3, 2) + expected_shape + (4,)
    with pytest.raises(ValueError):
        actual = random.multinomial(n, pvals, size=(1,) * 6)