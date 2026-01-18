import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_entropy_init(self):
    rg = Generator(self.bit_generator())
    rg2 = Generator(self.bit_generator())
    assert_(not comp_state(rg.bit_generator.state, rg2.bit_generator.state))