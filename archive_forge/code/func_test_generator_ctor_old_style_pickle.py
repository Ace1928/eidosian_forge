import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_generator_ctor_old_style_pickle():
    rg = np.random.Generator(np.random.PCG64DXSM(0))
    rg.standard_normal(1)
    ctor, args, state_a = rg.__reduce__()
    assert args[:1] == ('PCG64DXSM',)
    b = ctor(*args[:1])
    b.bit_generator.state = state_a
    state_b = b.bit_generator.state
    assert state_a == state_b