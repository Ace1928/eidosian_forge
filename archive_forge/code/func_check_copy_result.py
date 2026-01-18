import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def check_copy_result(x, y, ccontig, fcontig, strides=False):
    assert_(not x is y)
    assert_equal(x, y)
    assert_equal(res.flags.c_contiguous, ccontig)
    assert_equal(res.flags.f_contiguous, fcontig)