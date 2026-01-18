import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_broadcast_in_args(self):
    arrs = [np.empty((6, 7)), np.empty((5, 6, 1)), np.empty((7,)), np.empty((5, 1, 7))]
    mits = [np.broadcast(*arrs), np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])), np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])), np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])), np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1])]
    for mit in mits:
        assert_equal(mit.shape, (5, 6, 7))
        assert_equal(mit.ndim, 3)
        assert_equal(mit.nd, 3)
        assert_equal(mit.numiter, 4)
        for a, ia in zip(arrs, mit.iters):
            assert_(a is ia.base)