import math
import numpy as np
import cvxpy as cvx
from cvxpy.tests.base_test import BaseTest
def test_difference_kl_div_rel_entr(self) -> None:
    """A test showing the difference between kl_div and rel_entr
        """
    x = cvx.Variable()
    y = cvx.Variable()
    kl_div_prob = cvx.Problem(cvx.Minimize(cvx.kl_div(x, y)), constraints=[x + y <= 1])
    kl_div_prob.solve(solver=cvx.ECOS)
    self.assertItemsAlmostEqual(x.value, y.value)
    self.assertItemsAlmostEqual(kl_div_prob.value, 0)
    rel_entr_prob = cvx.Problem(cvx.Minimize(cvx.rel_entr(x, y)), constraints=[x + y <= 1])
    rel_entr_prob.solve(solver=cvx.ECOS)
    '\n        Reference solution computed by passing the following command to Wolfram Alpha:\n        minimize x*log(x/y) subject to {x + y <= 1, 0 <= x, 0 <= y}\n        '
    self.assertItemsAlmostEqual(x.value, 0.2178117, places=4)
    self.assertItemsAlmostEqual(y.value, 0.7821882, places=4)
    self.assertItemsAlmostEqual(rel_entr_prob.value, -0.278464)