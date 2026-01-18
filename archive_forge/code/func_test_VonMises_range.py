import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_VonMises_range(self):
    for mu in np.linspace(-7.0, 7.0, 5):
        r = random.vonmises(mu, 1, 50)
        assert_(np.all(r > -np.pi) and np.all(r <= np.pi))