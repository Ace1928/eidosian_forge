import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_broadcast_size_scalar():
    mu = np.ones(3)
    sigma = np.ones(3)
    random.normal(mu, sigma, size=3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=2)