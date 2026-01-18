import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
@unittest.skipIf(not glpk_available, 'GLPK not available')
def test_determine_valid_values(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var(RangeSet(4), domain=Binary)
    m.z = Var(domain=Integers, bounds=(-1, 2))
    m.constr = Constraint(expr=m.x == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
    m.logical = ConstraintList()
    m.logical.add(expr=m.y[1] + m.y[2] == 1)
    m.logical.add(expr=m.y[3] + m.y[4] == 1)
    m.logical.add(expr=m.y[2] + m.y[4] <= 1)
    var_to_values_map = determine_valid_values(m, detect_effectively_discrete_vars(m, 1e-06), Bunch(equality_tolerance=1e-06, pruning_solver='glpk'))
    valid_values = set([1, 2, 3, 4, 5])
    self.assertEqual(set(var_to_values_map[m.x]), valid_values)