import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.contrib.interior_point.inverse_reduced_hessian import (
def test_invrh_zavala_thesis(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    m.obj = pyo.Objective(expr=(m.x[1] - 1) ** 2 + (m.x[2] - 2) ** 2 + (m.x[3] - 3) ** 2)
    m.c1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3] == 0)
    status, invrh = inv_reduced_hessian_barrier(m, [m.x[2], m.x[3]])
    expected_invrh = np.asarray([[0.35714286, -0.21428571], [-0.21428571, 0.17857143]])
    np.testing.assert_array_almost_equal(invrh, expected_invrh)