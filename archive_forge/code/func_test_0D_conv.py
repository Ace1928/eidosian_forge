import numpy as np
import pytest
import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest
def test_0D_conv(self) -> None:
    """Convolution with 0D input.
        """
    for func in [cvx.conv, cvx.convolve]:
        x = cvx.Variable((1,))
        problem = cvx.Problem(cvx.Minimize(cvx.max(func(1.0, cvx.multiply(1.0, x)))), [x >= 0])
        problem.solve(cvx.ECOS)
        assert problem.status == cvx.OPTIMAL