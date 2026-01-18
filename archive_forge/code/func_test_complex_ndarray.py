import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_complex_ndarray(self) -> None:
    """Test ndarray of type complex64 and complex128.
        """
    x = Variable()
    z = np.full(1, 1j, dtype=np.complex64)
    x.value = 0
    expr = x + z
    assert np.isclose(expr.value, z)
    z = np.full(1, 1j, dtype=np.complex128)
    expr = x + z
    assert np.isclose(expr.value, z)