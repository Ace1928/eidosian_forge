import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_01_complex(self):
    ab = array([[0.0, 0.0, 2.0, 2.0], [-99, -1j, -1j, -1j], [4.0, 4.0, 4.0, 4.0]])
    b = array([2 - 1j, 4.0 - 1j, 4 + 1j, 2 + 1j])
    x = solveh_banded(ab, b)
    assert_array_almost_equal(x, [0.0, 1.0, 1.0, 0.0])