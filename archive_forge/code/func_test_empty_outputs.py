import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_empty_outputs(self):
    random = Generator(MT19937(self.seed))
    actual = random.multinomial(np.empty((10, 0, 6), 'i8'), [1 / 6] * 6)
    assert actual.shape == (10, 0, 6, 6)
    actual = random.multinomial(12, np.empty((10, 0, 10)))
    assert actual.shape == (10, 0, 10)
    actual = random.multinomial(np.empty((3, 0, 7), 'i8'), np.empty((3, 0, 7, 4)))
    assert actual.shape == (3, 0, 7, 4)