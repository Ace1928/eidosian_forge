import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
@unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
def test_logical_constraints_transformed(self):
    """It is expected that the result of this transformation is a MI(N)LP,
        so check that LogicalConstraints are handled correctly"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d2.c = Constraint()
    m.d = Disjunction(expr=[m.d1, m.d2])
    m.another = Disjunction(expr=[[m.x == 3], [m.x == 0]])
    m.Y = BooleanVar()
    m.global_logical = LogicalConstraint(expr=m.Y.xor(m.d1.indicator_var))
    m.d1.logical = LogicalConstraint(expr=implies(~m.Y, m.another.disjuncts[0].indicator_var))
    m.obj = Objective(expr=m.x)
    m.d1.indicator_var.set_value(True)
    m.d2.indicator_var.set_value(False)
    m.another.disjuncts[0].indicator_var.set_value(True)
    m.another.disjuncts[1].indicator_var.set_value(False)
    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    self.assertEqual(len(list(m.component_data_objects(LogicalConstraint, active=True, descend_into=(Block, Disjunct)))), 0)
    SolverFactory('gurobi').solve(m)
    self.assertTrue(value(m.d1.indicator_var))
    self.assertFalse(value(m.d2.indicator_var))
    self.assertTrue(value(m.another.disjuncts[0].indicator_var))
    self.assertFalse(value(m.another.disjuncts[1].indicator_var))
    self.assertEqual(value(m.Y.get_associated_binary()), 0)
    self.assertEqual(value(m.x), 3)