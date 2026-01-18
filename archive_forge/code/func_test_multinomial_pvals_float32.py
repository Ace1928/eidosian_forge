import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_multinomial_pvals_float32(self):
    x = np.array([0.99, 0.99, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09], dtype=np.float32)
    pvals = x / x.sum()
    match = '[\\w\\s]*pvals array is cast to 64-bit floating'
    with pytest.raises(ValueError, match=match):
        random.multinomial(1, pvals)