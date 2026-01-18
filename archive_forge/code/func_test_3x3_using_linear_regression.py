import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.contrib.interior_point.inverse_reduced_hessian import (
@unittest.skipIf(not numdiff_available, 'numdifftools missing')
@unittest.skipIf(not pandas_available, 'pandas missing')
def test_3x3_using_linear_regression(self):
    """simple linear regression with two x columns, so 3x3 Hessian"""
    model = self._simple_model()
    solver = pyo.SolverFactory('ipopt')
    status = solver.solve(model)
    self.assertTrue(check_optimal_termination(status))
    tstar = [pyo.value(model.b0), pyo.value(model.b['tofu']), pyo.value(model.b['chard'])]

    def _ndwrap(x):
        model.b0.fix(x[0])
        model.b['tofu'].fix(x[1])
        model.b['chard'].fix(x[2])
        rval = pyo.value(model.SSE)
        return rval
    H = nd.Hessian(_ndwrap)(tstar)
    HInv = np.linalg.inv(H)
    model.b0.fixed = False
    model.b['tofu'].fixed = False
    model.b['chard'].fixed = False
    status, H_inv_red_hess = inv_reduced_hessian_barrier(model, [model.b0, model.b['tofu'], model.b['chard']])
    np.testing.assert_array_almost_equal(HInv, H_inv_red_hess, decimal=3)