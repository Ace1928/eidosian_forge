import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
def test_hrectangular(self):
    self._test_common_lu_factor(self.hrect)