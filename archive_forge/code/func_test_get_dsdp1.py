import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.core.base.component import ComponentData
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
from pyomo.contrib.sensitivity_toolbox.sens import (
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_example
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import (
from pyomo.common.dependencies import scipy_available
@unittest.skipIf(not opt_kaug.available(False), 'k_aug is not available')
@unittest.skipIf(not opt_ipopt.available(False), 'ipopt is not available')
@unittest.skipIf(not scipy_available, 'scipy is not available')
def test_get_dsdp1(self):
    """
        It tests the function get_dsdp with a simple nonlinear programming example.

        min f: p1*x1+ p2*(x2^2) + p1*p2
         s.t c1: x1 = p1
             c2: x2 = p2
             c3: 10 <= p1 <= 10
             c4: 5 <= p2 <= 5
        """
    variable_name = ['p1', 'p2']
    m = ConcreteModel()
    m.x1 = Var(initialize=0)
    m.x2 = Var(initialize=0)
    m.p1 = Var(initialize=0)
    m.p2 = Var(initialize=0)
    m.obj = Objective(expr=m.x1 * m.p1 + m.x2 * m.x2 * m.p2 + m.p1 * m.p2, sense=minimize)
    m.c1 = Constraint(expr=m.x1 == m.p1)
    m.c2 = Constraint(expr=m.x2 == m.p2)
    theta = {'p1': 10.0, 'p2': 5.0}
    for v in variable_name:
        getattr(m, v).setlb(theta[v])
        getattr(m, v).setub(theta[v])
    dsdp, col = get_dsdp(m, variable_name, theta)
    ref = {'x1': [1.0, 0.0], 'x2': [0.0, 1.0], 'p1': [1.0, 0.0], 'p2': [0.0, 1.0]}
    np.testing.assert_almost_equal(dsdp.toarray(), np.vstack([ref[c] for c in col]).transpose())