import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_negative_binomial_invalid_p_n_combination(self):
    with np.errstate(invalid='ignore'):
        assert_raises(ValueError, random.negative_binomial, 2 ** 62, 0.1)
        assert_raises(ValueError, random.negative_binomial, [2 ** 62], [0.1])