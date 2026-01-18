import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_full_range_array(self, endpoint):
    for dt in self.itype:
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        ubnd = ubnd - 1 if endpoint else ubnd
        try:
            self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
        except Exception as e:
            raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))