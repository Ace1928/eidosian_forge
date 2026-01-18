import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_full_range(self):
    for dt in self.itype:
        lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
        ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
        try:
            self.rfunc(lbnd, ubnd, dtype=dt)
        except Exception as e:
            raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))