import pytest
from numpy.testing import assert_raises
from numpy import array_api as xp
import numpy as np
def test_isdtype_strictness():
    assert_raises(TypeError, lambda: xp.isdtype(xp.float64, 64))
    assert_raises(ValueError, lambda: xp.isdtype(xp.float64, 'f8'))
    assert_raises(TypeError, lambda: xp.isdtype(xp.float64, (('integral',),)))
    assert_raises(TypeError, lambda: xp.isdtype(xp.float64, np.object_))