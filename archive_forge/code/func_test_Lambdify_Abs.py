from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
@pytest.mark.parametrize('key', backends)
def test_Lambdify_Abs(key):
    if key == 'symengine':
        return
    be = Backend(key)
    x = be.Symbol('x')
    lmb = be.Lambdify([x], [be.Abs(x)])
    assert np.allclose([2], lmb([-2.0]))