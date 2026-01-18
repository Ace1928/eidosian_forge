import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_logitic(self):
    vals = self.rg.logistic(2.0, 2.0, 10)
    assert_(len(vals) == 10)