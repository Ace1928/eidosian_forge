import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_init_param_from_ndarray(self):
    m = ConcreteModel()
    m.ix_set = RangeSet(2)
    p_init = np.array([0, 5])

    def init_workaround(model, i):
        return p_init[i - 1]
    m.p = Param(m.ix_set, initialize=init_workaround)
    m.v = Var(m.ix_set)
    expr = m.p[1] > m.v[1]
    self.assertIsInstance(expr, InequalityExpression)
    self.assertEqual(str(expr), 'v[1]  <  0')
    expr = m.p[2] > m.v[2]
    self.assertIsInstance(expr, InequalityExpression)
    self.assertEqual(str(expr), 'v[2]  <  5')