from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_inactive_constraints(self):
    m = ConcreteModel()
    m.x = Var()
    m.c1 = Constraint(expr=m.x == 1)
    m.c2 = Constraint(expr=m.x == 2)
    m.o = Objective(expr=m.x)
    self.assertFalse(satisfiable(m))
    m.c2.deactivate()
    self.assertTrue(satisfiable(m))