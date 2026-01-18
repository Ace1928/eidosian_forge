from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose
def test_c_double(self):
    out = self.module.coddity.c_add(1, 2)
    exp_out = 3
    assert out == exp_out