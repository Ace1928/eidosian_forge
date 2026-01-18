import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def gen_vec(dtype):
    rand = np.random.default_rng()
    return rand.uniform(low=-1.0, high=1.0, size=(3,)).astype(dtype)