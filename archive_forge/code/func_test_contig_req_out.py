import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('dist', [random.standard_normal, random.random])
def test_contig_req_out(dist, order, dtype):
    out = np.empty((2, 3), dtype=dtype, order=order)
    variates = dist(out=out, dtype=dtype)
    assert variates is out
    variates = dist(out=out, dtype=dtype, size=out.shape)
    assert variates is out