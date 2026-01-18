import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def test_3d_power_cone_approx(self):
    """
        Use
            geo_mean((x,y), (alpha, 1-alpha)) >= |z|
        as a reformulation of
            PowCone3D(x, y, z, alpha).

        Check validity of the reformulation by solving
        orthogonal projection problems.
        """
    if 'MOSEK' in cp.installed_solvers():
        proj_solve_args = {'solver': 'MOSEK'}
    else:
        proj_solve_args = {'solver': 'SCS', 'eps': 1e-10}
    min_numerator = 2
    denominator = 25
    x = cp.Variable(3)
    np.random.seed(0)
    y = 10 * np.random.rand(3)
    for i, numerator in enumerate(range(min_numerator, denominator, 3)):
        alpha_float = numerator / denominator
        y[2] = y[0] ** alpha_float * y[1] ** (1 - alpha_float) + 0.05
        objective = cp.Minimize(cp.norm(y - x, 2))
        actual_constraints = [cp.constraints.PowCone3D(x[0], x[1], x[2], [alpha_float])]
        actual_prob = cp.Problem(objective, actual_constraints)
        actual_prob.solve(**proj_solve_args)
        actual_x = x.value.copy()
        weights = np.array([alpha_float, 1 - alpha_float])
        approx_constraints = [cp.geo_mean(x[:2], weights) >= cp.abs(x[2])]
        approx_prob = cp.Problem(objective, approx_constraints)
        approx_prob.solve()
        approx_x = x.value.copy()
        try:
            self.assertItemsAlmostEqual(actual_x, approx_x, places=4)
        except AssertionError as e:
            print(f'Failure at index {i} (when alpha={alpha_float}).')
            raise e