import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_error_byteorder(self):
    other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
    with pytest.raises(ValueError):
        random.integers(0, 200, size=10, dtype=other_byteord_dt)