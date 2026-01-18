import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_expr_construct_invalid(self):
    m = ConcreteModel()
    c = Constraint(rule=lambda m: None)
    self.assertRaisesRegex(ValueError, '.*rule returned None', m.add_component, 'c', c)
    m = ConcreteModel()
    c = Constraint([1], rule=lambda m, i: None)
    self.assertRaisesRegex(ValueError, '.*rule returned None', m.add_component, 'c', c)
    m = ConcreteModel()
    c = Constraint(rule=lambda m: True)
    self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(True\\).*Constraint\\.Feasible', m.add_component, 'c', c)
    m = ConcreteModel()
    c = Constraint([1], rule=lambda m, i: True)
    self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(True\\).*Constraint\\.Feasible', m.add_component, 'c', c)
    m = ConcreteModel()
    c = Constraint(rule=lambda m: False)
    self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(False\\).*Constraint\\.Infeasible', m.add_component, 'c', c)
    m = ConcreteModel()
    c = Constraint([1], rule=lambda m, i: False)
    self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(False\\).*Constraint\\.Infeasible', m.add_component, 'c', c)