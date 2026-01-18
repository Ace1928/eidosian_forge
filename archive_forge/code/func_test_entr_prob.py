import math
import numpy as np
import cvxpy as cvx
from cvxpy.tests.base_test import BaseTest
def test_entr_prob(self) -> None:
    """Test a problem with entr.
        """
    for n in [5, 10, 25]:
        print(n)
        x = cvx.Variable(n)
        obj = cvx.Maximize(cvx.sum(cvx.entr(x)))
        p = cvx.Problem(obj, [cvx.sum(x) == 1])
        p.solve(solver=cvx.ECOS, verbose=True)
        self.assertItemsAlmostEqual(x.value, n * [1.0 / n])
        p.solve(solver=cvx.SCS, verbose=True)
        self.assertItemsAlmostEqual(x.value, n * [1.0 / n], places=3)