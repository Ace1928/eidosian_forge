import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_invalid_pvals_broadcast(self):
    random = Generator(MT19937(self.seed))
    pvals = [[1 / 6] * 6, [1 / 4] * 6]
    assert_raises(ValueError, random.multinomial, 1, pvals)
    assert_raises(ValueError, random.multinomial, 6, 0.5)