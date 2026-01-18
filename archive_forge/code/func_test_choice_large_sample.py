import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_choice_large_sample(self):
    choice_hash = '4266599d12bfcfb815213303432341c06b4349f5455890446578877bb322e222'
    random = Generator(MT19937(self.seed))
    actual = random.choice(10000, 5000, replace=False)
    if sys.byteorder != 'little':
        actual = actual.byteswap()
    res = hashlib.sha256(actual.view(np.int8)).hexdigest()
    assert_(choice_hash == res)