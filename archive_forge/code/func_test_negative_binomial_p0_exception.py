import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_negative_binomial_p0_exception(self):
    with assert_raises(ValueError):
        x = random.negative_binomial(1, 0)