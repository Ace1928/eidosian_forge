import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_bounds_checking_array(self, endpoint):
    for dt in self.itype:
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)
        assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2, endpoint=endpoint, dtype=dt)
        assert_raises(ValueError, self.rfunc, [lbnd] * 2, [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
        assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2, endpoint=endpoint, dtype=dt)
        assert_raises(ValueError, self.rfunc, [1] * 2, 0, endpoint=endpoint, dtype=dt)