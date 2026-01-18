from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
@pytest.mark.parametrize('key', backends)
def test_Lambdify_sign(key):
    if key.endswith('symengine'):
        return
    be = Backend(key)
    x = be.Symbol('x')
    lmb = be.Lambdify([x], [be.sign(x)])
    assert np.allclose([1], lmb([3.0]))
    assert np.allclose([-1], lmb([-2.0]))
    assert np.allclose([0], lmb([0.0]))