import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_gamma_floats(self):
    rg = Generator(self.bit_generator())
    warmup(rg)
    state = rg.bit_generator.state
    r1 = rg.standard_gamma(4.0, 11, dtype=np.float32)
    rg2 = Generator(self.bit_generator())
    warmup(rg2)
    rg2.bit_generator.state = state
    r2 = rg2.standard_gamma(4.0, 11, dtype=np.float32)
    assert_array_equal(r1, r2)
    assert_equal(r1.dtype, np.float32)
    assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))