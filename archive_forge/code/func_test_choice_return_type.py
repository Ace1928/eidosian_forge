import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_choice_return_type(self):
    p = np.ones(4) / 4.0
    actual = random.choice(4, 2)
    assert actual.dtype == np.int64
    actual = random.choice(4, 2, replace=False)
    assert actual.dtype == np.int64
    actual = random.choice(4, 2, p=p)
    assert actual.dtype == np.int64
    actual = random.choice(4, 2, p=p, replace=False)
    assert actual.dtype == np.int64